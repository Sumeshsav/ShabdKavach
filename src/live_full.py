"""
live_full.py - the whole system, live: two microphones, LMS, and the
trained MaskNet.

TWO POSSIBLE ORDERS, and it is worth testing both.

  --order lms-first     mic -> LMS -> neural -> out
      The adaptive filter removes whatever noise is linearly related to
      the reference mic. The network then cleans up what is left. The
      network sees an easier problem.

  --order nn-first      mic -> neural -> LMS -> out
      This is the order in our architecture diagram: the network does
      the heavy lifting, the classical filter trims residual noise.
      Harder for the LMS, because after masking the leftover noise is
      no longer a simple linear function of the reference.

TIMING SUBTLETY (this bites if you get it wrong)
The neural stage delays audio by n_fft - hop = 768 samples (16 ms). In
nn-first order its output therefore lags the reference microphone by
that much. An adaptive filter fed with misaligned signals cannot
converge, so the reference is passed through a matching delay line.
In lms-first order both signals are still aligned at the input, so no
delay is needed.

USAGE
  python live_full.py --device-in 15 --ckpt ../checkpoints/best37.pt
  python live_full.py --device-in 15 --ckpt ../checkpoints/best37.pt \\
                      --order nn-first
  python live_full.py --device-in 15 --ckpt ../checkpoints/best37.pt \\
                      --no-lms                 # model only, for A/B
  python live_full.py --device-in 15 --bypass  # raw mic, for A/B
"""

import argparse
import time

import numpy as np
import torch
import sounddevice as sd

from model import MaskNet
from chain import EnhancementChain

SR = 48000


class BlockNLMS:
    """Block normalised LMS - one update per block, vectorised."""

    def __init__(self, taps=128, mu=0.02, eps=1e-6, wmax=8.0):
        self.taps, self.mu, self.eps, self.wmax = taps, mu, eps, wmax
        self.reset()

    def reset(self):
        self.w = np.zeros(self.taps, dtype=np.float32)
        self.hist = np.zeros(self.taps - 1, dtype=np.float32)

    def __call__(self, primary, reference, adapt=True):
        n = len(primary)
        ext = np.concatenate([self.hist, reference])
        X = np.lib.stride_tricks.sliding_window_view(ext, self.taps)[:n]
        X = X[:, ::-1]
        e = primary - X @ self.w
        if adapt:
            p = float(np.sum(X * X)) / n + self.eps
            self.w += (self.mu / p) * (X.T @ e)
            nw = float(np.linalg.norm(self.w))
            if nw > self.wmax:
                self.w *= self.wmax / nw
        self.hist = ext[-(self.taps - 1):]
        return e.astype(np.float32)


class Delay:
    """Fixed-length delay line, so the reference stays time-aligned with
    the neural stage's output."""

    def __init__(self, samples):
        self.n = int(samples)
        self.buf = np.zeros(self.n, dtype=np.float32) if self.n > 0 else None

    def __call__(self, x):
        if self.n <= 0:
            return x
        out = np.concatenate([self.buf, x])[:len(x)]
        self.buf = np.concatenate([self.buf, x])[len(x):]
        return out.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--device-in", type=int)
    p.add_argument("--device-out", type=int, default=None)
    p.add_argument("--ckpt")
    p.add_argument("--primary-ch", type=int, default=0)
    p.add_argument("--ref-ch", type=int, default=1)
    p.add_argument("--order", choices=["lms-first", "nn-first"],
                   default="lms-first")
    p.add_argument("--hop", type=int, default=512)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--taps", type=int, default=128)
    p.add_argument("--mu", type=float, default=0.02)
    p.add_argument("--mask-floor", type=float, default=0.15)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--no-lms", action="store_true")
    p.add_argument("--bypass", action="store_true")
    args = p.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    info = sd.query_devices(args.device_in)
    print(f"input : {info['name']}  ({info['max_input_channels']} ch)")
    if info["max_input_channels"] < 2:
        print("Need a 2-channel input device.")
        return

    chain = None
    if not args.bypass:
        ck = torch.load(args.ckpt, map_location="cpu")
        a = ck.get("args", {})
        m = MaskNet(hidden=a.get("hidden", 256), layers=a.get("layers", 2))
        m.load_state_dict(ck["model"])
        print(f"model : epoch {ck.get('epoch')}, "
              f"val SI-SNR {ck.get('val_si_snr', float('nan')):.2f} dB")
        # the two extra chain stages are evaluated separately; here we
        # want the model plus the mask floor only
        chain = EnhancementChain(m, args.n_fft, args.hop,
                                 mask_floor=args.mask_floor,
                                 snr_aware=False, lms=False)
        for _ in range(20):                      # prime, hides lazy init
            chain.process_block(np.zeros(args.hop, dtype=np.float32))

    lms = None if (args.no_lms or args.bypass) else BlockNLMS(args.taps,
                                                              args.mu)
    delay = Delay(chain.latency_samples
                  if (chain and args.order == "nn-first" and lms) else 0)

    mode = ("BYPASS - raw primary mic" if args.bypass
            else ("model only" if args.no_lms
                  else f"{args.order}  (LMS {args.taps} taps, mu {args.mu})"))
    print(f"primary = ch{args.primary_ch}   reference = ch{args.ref_ch}")
    print(f"block {args.hop} ({1000*args.hop/SR:.1f} ms), mode: {mode}")
    if chain:
        print(f"neural latency {chain.latency_ms:.1f} ms"
              + (f", reference delayed to match" if delay.n else ""))
    print("Ctrl+C to stop.\n")

    st = {"n": 0, "in": 0.0, "out": 0.0, "worst": 0.0}

    def cb(indata, outdata, frames, t, status):
        if status:
            print(status)
        prim = indata[:, args.primary_ch].astype(np.float32)
        ref = indata[:, args.ref_ch].astype(np.float32)
        t0 = time.perf_counter()

        if args.bypass:
            out = prim
        elif lms is None:                        # model only
            out = chain.process_block(prim)
        elif args.order == "lms-first":
            out = chain.process_block(lms(prim, ref))
        else:                                    # nn-first
            out = lms(chain.process_block(prim), delay(ref))

        dt = (time.perf_counter() - t0) * 1000
        outdata[:] = np.repeat((out * args.gain)[:, None],
                               outdata.shape[1], axis=1)
        st["n"] += 1
        st["in"] += float(np.mean(prim ** 2))
        st["out"] += float(np.mean(out ** 2))
        st["worst"] = max(st["worst"], dt)

    try:
        with sd.Stream(samplerate=SR, blocksize=args.hop,
                       device=(args.device_in, args.device_out),
                       channels=(2, 2), dtype="float32", callback=cb):
            while True:
                time.sleep(1.0)
                if st["n"]:
                    ri = np.sqrt(st["in"] / st["n"])
                    ro = np.sqrt(st["out"] / st["n"])
                    d = 20 * np.log10((ro + 1e-9) / (ri + 1e-9))
                    line = (f"in {ri:.4f}  out {ro:.4f}  ({d:+.1f} dB)   "
                            f"worst {st['worst']:.2f} ms "
                            f"/ {1000*args.hop/SR:.1f} ms")
                    if lms is not None:
                        line += f"   ||w|| {np.linalg.norm(lms.w):.3f}"
                    print(line)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
