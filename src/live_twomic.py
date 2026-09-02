"""
live_twomic.py - two-microphone LMS, running live.

WHY A DIFFERENT IMPLEMENTATION
two_mic_lms.py updates the filter once per SAMPLE. That is the textbook
form and it is fine offline, but 48000 Python-loop iterations per second
cannot fit in a 5.33 ms audio callback.

This uses BLOCK LMS instead: filter the whole block at once with the
current weights, then apply one update from the whole block's error.
Same algorithm, vectorised, roughly a hundred times faster, and it
converges to the same place.

    y = X w                 X holds the sliding reference windows
    e = d - y               d is the primary mic, e is the output
    w += mu X^T e / (power) one update per block

WIRING
    ch0 / ch1 come from a 2-channel input device.
    PRIMARY   should be the mic nearer your mouth.
    REFERENCE should be the mic nearer the noise.
    If it sounds wrong, swap --primary-ch and --ref-ch.

USAGE
  python live_twomic.py --list
  python live_twomic.py --device-in 15
  python live_twomic.py --device-in 15 --bypass        # A/B against raw
  python live_twomic.py --device-in 15 --primary-ch 1 --ref-ch 0
"""

import argparse
import time

import numpy as np
import sounddevice as sd

SR = 48000


class BlockNLMS:
    """Block normalised LMS. Filter state persists across blocks."""

    def __init__(self, taps=128, mu=0.02, eps=1e-6, wmax=8.0):
        self.taps = taps
        self.mu = mu
        self.eps = eps
        self.wmax = wmax
        self.reset()

    def reset(self):
        self.w = np.zeros(self.taps, dtype=np.float32)
        # history of past reference samples, needed because the filter
        # window at the start of a block reaches back into the previous
        self.hist = np.zeros(self.taps - 1, dtype=np.float32)

    def __call__(self, primary, reference, adapt=True):
        n = len(primary)
        ext = np.concatenate([self.hist, reference])
        # X[i] = ref[i], ref[i-1], ... ref[i-taps+1]
        X = np.lib.stride_tricks.sliding_window_view(ext, self.taps)[:n]
        X = X[:, ::-1]

        y = X @ self.w
        e = primary - y

        if adapt:
            p = float(np.sum(X * X)) / n + self.eps
            self.w += (self.mu / p) * (X.T @ e)
            nw = float(np.linalg.norm(self.w))
            if nw > self.wmax:
                self.w *= self.wmax / nw

        self.hist = ext[-(self.taps - 1):]
        return e.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--device-in", type=int)
    p.add_argument("--device-out", type=int, default=None)
    p.add_argument("--primary-ch", type=int, default=0)
    p.add_argument("--ref-ch", type=int, default=1)
    p.add_argument("--taps", type=int, default=128)
    p.add_argument("--mu", type=float, default=0.02)
    p.add_argument("--block", type=int, default=512)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--bypass", action="store_true",
                   help="pass the primary mic straight through, for A/B")
    args = p.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    info = sd.query_devices(args.device_in)
    print(f"input : {info['name']}  ({info['max_input_channels']} ch)")
    if info["max_input_channels"] < 2:
        print("Need a 2-channel input device.")
        return
    print(f"primary = ch{args.primary_ch}   reference = ch{args.ref_ch}")
    print(f"block {args.block} samples ({1000*args.block/SR:.1f} ms), "
          f"{args.taps} taps, mu {args.mu}")
    print("BYPASS - raw primary mic" if args.bypass
          else "LMS active")
    print("Ctrl+C to stop.\n")

    f = BlockNLMS(args.taps, args.mu)
    stats = {"n": 0, "in": 0.0, "out": 0.0, "worst": 0.0}

    def cb(indata, outdata, frames, t, status):
        if status:
            print(status)
        prim = indata[:, args.primary_ch].astype(np.float32)
        ref = indata[:, args.ref_ch].astype(np.float32)
        t0 = time.perf_counter()
        out = prim if args.bypass else f(prim, ref)
        dt = (time.perf_counter() - t0) * 1000
        outdata[:] = np.repeat((out * args.gain)[:, None],
                               outdata.shape[1], axis=1)
        stats["n"] += 1
        stats["in"] += float(np.mean(prim ** 2))
        stats["out"] += float(np.mean(out ** 2))
        stats["worst"] = max(stats["worst"], dt)

    dev = (args.device_in, args.device_out)
    try:
        with sd.Stream(samplerate=SR, blocksize=args.block, device=dev,
                       channels=(2, 2), dtype="float32", callback=cb):
            while True:
                time.sleep(1.0)
                if stats["n"]:
                    ri = np.sqrt(stats["in"] / stats["n"])
                    ro = np.sqrt(stats["out"] / stats["n"])
                    d = 20 * np.log10((ro + 1e-9) / (ri + 1e-9))
                    print(f"in {ri:.4f}  out {ro:.4f}  ({d:+.1f} dB)   "
                          f"||w|| {np.linalg.norm(f.w):.3f}   "
                          f"worst block {stats['worst']:.2f} ms "
                          f"/ {1000*args.block/SR:.1f} ms budget")
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
