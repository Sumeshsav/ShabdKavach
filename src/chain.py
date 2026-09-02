"""
chain.py - THE DEPLOYMENT ENTRY POINT.

Running this with no stage flags gives you exactly what ships:

    python chain.py --live --ckpt ../checkpoints/best37.pt

DEFAULT (deployment) CONFIGURATION
    mask floor   ON   (0.15)   bounded guarantee, costs almost nothing
    SNR-aware    OFF           estimator is not yet trustworthy
    LMS trim     OFF           no useful range on a single microphone

The two disabled stages are kept behind --snr-aware and --lms so they
can still be ablated and measured. They are NOT on by default, because
shipping a feature you have not validated is worse than not shipping it.

WHY SNR-AWARE IS OFF
The mask-derived SNR estimate reported -10.1 dB on a file mixed at
+10 dB. It measures how confident the mask is, not how clean the input
is, so the blend it drives is not trustworthy. A minimum-statistics
noise-floor tracker would fix it; until that is built and measured, the
stage stays off.

WHY LMS IS OFF
Measured on the trained checkpoint: below a 0.08 magnitude clamp it
changes nothing (-0.43 dB energy, adapting on 4% of frames); above it,
it cancels speech. The mask-derived reference is not speech-free. A
real second microphone fixes this - see two_mic_lms.py, which measured
+7.67 dB SI-SNR with a genuinely clean reference.

All stages are INFERENCE-TIME. They need no retraining and work with
the existing epoch-37 checkpoint unchanged.

  0. INPUT GAIN NORMALISATION  (always on)
     Training data was peak-normalised to 0.9 by prep_data.py, so the
     network only ever saw audio in that range. A quiet recording -
     measured at 0.12 peak - falls outside it, log1p(magnitude) lands
     somewhere the GRU was never trained, and the predicted mask is
     meaningless. We measured -9.87 dB SI-SNR on such an input.

     Fix: track a slow running peak, scale the frame up to a reference
     level before the STFT, and undo exactly that scale on the output.
     The listener hears the original loudness; the network always sees
     the level it was trained on. The gain is smoothed so it cannot
     pump between blocks, and it is bounded so near-silence is not
     amplified into noise.

  1. MASK FLOOR      mask.clamp(min=floor)
     Bounds worst-case attenuation by construction. With floor = 0.15 no
     frequency bin can lose more than 20*log10(0.15) = -16.5 dB, so
     speech can be degraded but never deleted. No detector to be wrong.

  2. SNR-AWARE BLEND
     We measured DeepFilterNet3 degrading audio above 5 dB input SNR
     (SI-SNR 10.01 -> 5.46). Cause: processing distortion is roughly
     constant while the noise available to remove shrinks. So we
     estimate SNR from the mask itself and blend the enhanced spectrum
     back toward the input as the input gets cleaner.

  3. LMS TRIM (single microphone)
     Classical NLMS needs a noise-only reference channel, which a
     one-mic headset does not have. But the mask already splits the
     spectrum: X*m is the speech estimate and X*(1-m) is the noise
     estimate. We use that noise estimate as the NLMS reference and
     adapt one complex tap per frequency bin to cancel residual noise
     left in the speech estimate.
     NOTE: mu, wmax and gate MUST be tuned on your own checkpoint. If
     the mask is not discriminative the reference still contains speech
     and the filter will cancel it. Watch the "LMS energy change"
     diagnostic - beyond about -3 dB it is eating speech, so lower
     --lms-wmax or disable the stage with --no-lms.

     Adaptation is FROZEN on speech-dominant frames, so the filter
     cannot learn to cancel the speech itself. The gate is relative to
     a running average of the mask, not a fixed threshold, so it
     self-calibrates to whatever the model outputs.

USAGE
  python chain.py --check                        # reconstruction test
  python chain.py --live --ckpt best37.pt        # DEPLOYMENT
  python chain.py --infile in.wav --outfile out.wav --ckpt best37.pt

  # ablation - measure what each stage costs
  python chain.py ... --mask-floor 0.0           # no guarantee
  python chain.py ... --snr-aware                # enable stage 2
  python chain.py ... --lms --lms-wmax 0.08      # enable stage 3
"""

import argparse
import time

import numpy as np
import torch

from model import MaskNet

SR = 48000


# ----------------------------------------------------------------------
class EnhancementChain:
    """MaskNet + mask floor + SNR-aware blend + per-bin NLMS trim.

    Streaming: call process_block() with exactly `hop` samples and get
    `hop` samples back. All state persists between calls.
    """

    def __init__(self, model, n_fft=1024, hop=256,
                 mask_floor=0.15,
                 snr_aware=False, snr_low=0.0, snr_high=12.0, alpha_min=0.30,
                 snr_smooth=0.95,
                 lms=False, lms_mu=0.01, lms_gate=1.00, lms_wmax=0.30,
                 agc=True, agc_target=0.30, agc_smooth=0.995,
                 agc_max_gain=20.0, agc_floor=1e-4,
                 device="cpu"):
        self.model = model.eval().to(device)
        self.device = device
        self.n_fft, self.hop = n_fft, hop
        self.n_bins = n_fft // 2 + 1

        self.mask_floor = float(mask_floor)
        self.snr_aware = snr_aware
        self.snr_low, self.snr_high = snr_low, snr_high
        self.alpha_min, self.snr_smooth = alpha_min, snr_smooth
        self.lms = lms
        self.lms_mu, self.lms_gate, self.lms_wmax = lms_mu, lms_gate, lms_wmax
        self.agc = agc
        self.agc_target = agc_target
        self.agc_smooth = agc_smooth
        self.agc_max_gain = agc_max_gain
        self.agc_floor = agc_floor

        self.window = torch.hann_window(n_fft)
        w2 = self.window.numpy() ** 2
        norm = np.zeros(n_fft)
        for k in range(0, n_fft, hop):
            norm += np.roll(w2, k)
        self.wola_norm = float(np.median(norm))

        self.reset()

    # ---- state ------------------------------------------------------
    def reset(self):
        self.h = None                                   # GRU hidden state
        self.in_buf = np.zeros(self.n_fft, dtype=np.float32)
        self.out_buf = np.zeros(self.n_fft, dtype=np.float32)
        self.w = torch.zeros(self.n_bins, dtype=torch.complex64)  # NLMS taps
        self.snr_db = 0.0                               # smoothed estimate
        self.mask_ema = None                            # running mask level
        self.peak_ema = None                            # running input peak
        self.gain = 1.0                                 # current AGC gain
        self.log = {"snr": [], "alpha": [], "mask_mean": [], "adapting": 0,
                    "e_pre": 0.0, "e_post": 0.0}

    # ---- stage 2 helper ---------------------------------------------
    def _estimate_snr(self, mag, mask):
        """A-posteriori SNR from the mask, in dB.

        speech power = sum |X|^2 m^2 ; noise power = sum |X|^2 (1-m)^2
        No clean reference exists at runtime, so this is non-intrusive
        by necessity.
        """
        p = mag ** 2
        s = float(torch.sum(p * mask ** 2))
        n = float(torch.sum(p * (1.0 - mask) ** 2))
        inst = 10.0 * np.log10((s + 1e-10) / (n + 1e-10))
        inst = float(np.clip(inst, -20.0, 40.0))
        a = self.snr_smooth
        self.snr_db = a * self.snr_db + (1 - a) * inst
        return self.snr_db

    def _alpha(self, snr_db):
        """1.0 = process fully, alpha_min = mostly pass the input through."""
        span = self.snr_high - self.snr_low
        a = (self.snr_high - snr_db) / (span + 1e-9)
        return float(np.clip(a, self.alpha_min, 1.0))

    # ---- one block ---------------------------------------------------
    @torch.no_grad()
    def process_block(self, block):
        assert len(block) == self.hop, f"expected {self.hop} samples"

        # 1. slide new samples into the analysis buffer
        self.in_buf[:-self.hop] = self.in_buf[self.hop:]
        self.in_buf[-self.hop:] = block

        # ---- STAGE 0 : input gain normalisation ---------------------
        # Bring the frame to the level the network was trained on, and
        # remember the gain so it can be undone on the way out.
        g = 1.0
        if self.agc:
            pk = float(np.max(np.abs(self.in_buf)))
            if self.peak_ema is None:
                self.peak_ema = pk
            else:
                a = self.agc_smooth
                # follow a rising peak immediately, fall back slowly, so
                # a loud burst cannot be clipped and a pause cannot pump
                self.peak_ema = (max(pk, a * self.peak_ema + (1 - a) * pk)
                                 if pk > self.peak_ema
                                 else a * self.peak_ema + (1 - a) * pk)
            if self.peak_ema > self.agc_floor:
                g = min(self.agc_target / self.peak_ema, self.agc_max_gain)
            self.gain = g

        # 2. analysis
        frame = torch.from_numpy(self.in_buf.copy() * g) * self.window
        spec = torch.fft.rfft(frame)
        mag = spec.abs()

        # 3. one causal GRU step, hidden state carried forward
        feat = torch.log1p(mag).view(1, 1, -1).to(self.device)
        out, self.h = self.model.gru(feat, self.h)
        mask = torch.sigmoid(self.model.out(out)).view(-1).cpu()

        # ---- STAGE 1 : mask floor -----------------------------------
        if self.mask_floor > 0:
            mask = mask.clamp(min=self.mask_floor)

        enhanced = spec * mask

        # ---- STAGE 2 : SNR-aware blend ------------------------------
        alpha = 1.0
        if self.snr_aware:
            snr = self._estimate_snr(mag, mask)
            alpha = self._alpha(snr)
            enhanced = alpha * enhanced + (1.0 - alpha) * spec
            self.log["snr"].append(snr)
        self.log["alpha"].append(alpha)

        # ---- STAGE 3 : per-bin NLMS trim ----------------------------
        # reference = the model's own noise estimate
        if self.lms:
            self.log["e_pre"] += float(torch.sum(enhanced.abs() ** 2))
            ref = spec * (1.0 - mask)
            y = self.w * ref
            err = enhanced - y
            mask_mean = float(mask.mean())
            self.log["mask_mean"].append(mask_mean)
            # Self-calibrating gate: adapt only on frames the model
            # considers NOISIER THAN USUAL. A fixed threshold breaks as
            # soon as the model's typical mask level changes.
            if self.mask_ema is None:
                self.mask_ema = mask_mean
            else:
                self.mask_ema = 0.98 * self.mask_ema + 0.02 * mask_mean
            if mask_mean < self.lms_gate * self.mask_ema:
                p = (ref.abs() ** 2) + 1e-8
                self.w = self.w + self.lms_mu * torch.conj(ref) * err / p
                mag_w = self.w.abs()
                over = mag_w > self.lms_wmax
                if bool(over.any()):
                    self.w[over] = self.w[over] / mag_w[over] * self.lms_wmax
                self.log["adapting"] += 1
            enhanced = err
            self.log["e_post"] += float(torch.sum(enhanced.abs() ** 2))

        # 4. synthesis - undo the input gain so the listener hears the
        #    original loudness
        rec = torch.fft.irfft(enhanced, n=self.n_fft)
        rec = (rec * self.window).numpy() / g

        # 5. overlap-add
        self.out_buf += rec / self.wola_norm
        output = self.out_buf[:self.hop].copy()
        self.out_buf[:-self.hop] = self.out_buf[self.hop:]
        self.out_buf[-self.hop:] = 0.0
        return output.astype(np.float32)

    def process_signal(self, audio):
        """Batch convenience wrapper - identical code path to streaming."""
        nb = len(audio) // self.hop
        out = np.zeros(nb * self.hop, dtype=np.float32)
        for i in range(nb):
            out[i * self.hop:(i + 1) * self.hop] = \
                self.process_block(audio[i * self.hop:(i + 1) * self.hop])
        return out

    @property
    def latency_samples(self):
        return self.n_fft - self.hop

    @property
    def latency_ms(self):
        return 1000.0 * self.latency_samples / SR


# ----------------------------------------------------------------------
def run_check(n_fft=1024, hop=256):
    """With every stage disabled and the mask forced to 1.0 the chain must
    reconstruct the input exactly. Proves the added stages did not break
    the analysis-synthesis path."""
    print("Reconstruction check - all stages OFF, mask forced to 1.0\n")

    class Identity(torch.nn.Module):
        def __init__(self, nb):
            super().__init__()
            self.gru = lambda f, h: (f, h)
            self.out = lambda x: torch.full((*x.shape[:-1], nb), 20.0)

    ch = EnhancementChain(Identity(n_fft // 2 + 1), n_fft, hop,
                          mask_floor=0.0, snr_aware=False, lms=False)
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2, 2 * SR, endpoint=False)
    sig = (0.3 * np.sin(2 * np.pi * 220 * t)
           + 0.1 * rng.standard_normal(len(t))).astype(np.float32)
    out = ch.process_signal(sig)
    d = ch.latency_samples
    a, b = sig[:len(out) - d], out[d:]
    err = np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a ** 2)) + 1e-12)
    print(f"delay           : {d} samples ({ch.latency_ms:.1f} ms)")
    print(f"relative RMS err: {err:.6f}")
    print("PASS" if err < 1e-4 else "FAIL")


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    a = ck.get("args", {})
    m = MaskNet(hidden=a.get("hidden", 256), layers=a.get("layers", 2))
    m.load_state_dict(ck["model"])
    print(f"loaded {path} (epoch {ck.get('epoch')}, "
          f"val SI-SNR {ck.get('val_si_snr', float('nan')):.2f} dB)")
    return m


def build(args, model):
    return EnhancementChain(
        model, args.n_fft, args.hop,
        mask_floor=args.mask_floor,
        agc=not args.no_agc, agc_target=args.agc_target,
        snr_aware=args.snr_aware,
        snr_low=args.snr_low, snr_high=args.snr_high,
        alpha_min=args.alpha_min,
        lms=args.lms, lms_mu=args.lms_mu, lms_gate=args.lms_gate,
        lms_wmax=args.lms_wmax)


def run_file(args, model):
    import soundfile as sf
    import librosa
    audio, _ = librosa.load(args.infile, sr=SR, mono=True)
    ch = build(args, model)

    worst, t0 = 0.0, time.time()
    nb = len(audio) // args.hop
    out = np.zeros(nb * args.hop, dtype=np.float32)
    for i in range(nb):
        s = time.perf_counter()
        out[i * args.hop:(i + 1) * args.hop] = \
            ch.process_block(audio[i * args.hop:(i + 1) * args.hop])
        worst = max(worst, (time.perf_counter() - s) * 1000)
    total = time.time() - t0
    sf.write(args.outfile, out, SR)

    budget = 1000.0 * args.hop / SR
    print(f"\nstages          : agc={not args.no_agc} "
          f"floor={args.mask_floor} "
          f"snr_aware={args.snr_aware} lms={args.lms}")
    if not args.no_agc:
        print(f"input gain      : x{ch.gain:.1f} at end of file "
              f"(target peak {args.agc_target})")
    print(f"blocks          : {nb}")
    print(f"block budget    : {budget:.2f} ms")
    print(f"mean per block  : {1000*total/nb:.2f} ms")
    print(f"worst block     : {worst:.2f} ms")
    print(f"RTF             : {total/(len(out)/SR):.3f}")
    print(f"latency         : {ch.latency_ms:.1f} ms")
    if ch.log["snr"]:
        print(f"est. input SNR  : {np.mean(ch.log['snr']):+.1f} dB mean")
        print(f"blend alpha     : {np.mean(ch.log['alpha']):.2f} mean "
              f"(1.0 = full processing)")
    if args.lms:
        pre, post = ch.log["e_pre"], ch.log["e_post"]
        d = 10 * np.log10((post + 1e-12) / (pre + 1e-12))
        print(f"NLMS adapting   : {100*ch.log['adapting']/max(nb,1):.0f}% "
              f"of frames,  |w| max {float(ch.w.abs().max()):.3f} "
              f"(clamp {ch.lms_wmax})")
        print(f"LMS energy chg  : {d:+.2f} dB"
              + ("   <-- likely eating speech, lower --lms-wmax"
                 if d < -3.0 else ""))
    print(f"in rms {np.sqrt(np.mean(audio[:len(out)]**2)):.5f}  ->  "
          f"out rms {np.sqrt(np.mean(out**2)):.5f}")
    print(f"wrote {args.outfile}")


def run_live(args, model):
    import sounddevice as sd
    ch = build(args, model)
    for _ in range(20):                      # prime, hides lazy-init glitch
        ch.process_block(np.zeros(args.hop, dtype=np.float32))
    print(f"live: block {1000*args.hop/SR:.2f} ms, latency "
          f"{ch.latency_ms:.1f} ms. Ctrl+C to stop.")

    def cb(indata, outdata, frames, t, status):
        if status:
            print(status)
        outdata[:, 0] = ch.process_block(indata[:, 0].copy()) * args.gain

    try:
        with sd.Stream(samplerate=SR, blocksize=args.hop, channels=1,
                       dtype="float32", callback=cb):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt")
    p.add_argument("--infile"); p.add_argument("--outfile", default="chain.wav")
    p.add_argument("--live", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--gain", type=float, default=1.0)
    # stage controls, so every stage can be ablated
    p.add_argument("--mask-floor", type=float, default=0.15)
    p.add_argument("--no-agc", action="store_true",
                   help="disable input gain normalisation (on by default)")
    p.add_argument("--agc-target", type=float, default=0.30,
                   help="peak level the network is fed")
    p.add_argument("--snr-aware", action="store_true",
                   help="ENABLE the SNR-aware blend (off by default; the "
                        "estimator is not yet trustworthy)")
    p.add_argument("--snr-low", type=float, default=0.0)
    p.add_argument("--snr-high", type=float, default=12.0)
    p.add_argument("--alpha-min", type=float, default=0.30)
    p.add_argument("--lms", action="store_true",
                   help="ENABLE the single-mic LMS trim (off by default; no "
                        "useful operating range on one microphone)")
    p.add_argument("--lms-mu", type=float, default=0.05)
    p.add_argument("--lms-gate", type=float, default=1.00,
                   help="adapt when frame mask < gate x running mean mask; "
                        "lower = stricter, adapts on fewer frames")
    p.add_argument("--lms-wmax", type=float, default=0.30,
                   help="max filter magnitude per bin; lower = less "
                        "cancellation. Drop this first if speech vanishes")
    args = p.parse_args()

    if args.check:
        run_check(args.n_fft, args.hop)
        raise SystemExit

    model = load_model(args.ckpt)
    if args.live:
        run_live(args, model)
    else:
        run_file(args, model)
