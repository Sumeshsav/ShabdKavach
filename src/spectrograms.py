"""
spectrograms.py - the single most persuasive figure in your deck.

Produces one row of spectrograms on a SHARED dB scale:
    clean reference | noisy input | DeepFilterNet3 | our model

WHY THE STREAMING PATH
Our model panel is produced by chain.py's streaming engine, not by
model.forward() on the whole clip. Whole-clip processing runs one STFT
over the entire file and the GRU from start to finish - which is NOT
what runs on hardware and cannot run live at all. Using the streaming
path means the figure shows what actually ships.

WHY A SHARED SCALE MATTERS
baseline.py normalises each panel with ref=np.max, so every panel gets
its own reference level. That makes panels look different when they are
not, and it is technically misleading in a comparison figure. Here every
panel uses the SAME reference (peak of the clean signal), so brightness
means the same thing everywhere and differences you see are real.

USAGE
  python spectrograms.py --clean ../data/speech_test/x.wav \\
                         --noise ../data/noise_test/impulse_gunshots.wav \\
                         --snr 0 --ckpt ../checkpoints/best.pt

  # skip DeepFilterNet if not installed here
  python spectrograms.py ... --skip-dfn
"""

import argparse
import os

import numpy as np
import librosa
import librosa.display
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import mix_at_snr
from model import MaskNet
from chain import EnhancementChain

SR = 48000

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "savefig.dpi": 200,
})


def si_snr(ref, est, eps=1e-8):
    """Scale-invariant SNR in dB. Higher is better; immune to volume."""
    n = min(len(ref), len(est))
    ref, est = ref[:n].astype(np.float64), est[:n].astype(np.float64)
    ref, est = ref - ref.mean(), est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    tgt = a * ref
    return 10 * np.log10((np.sum(tgt ** 2) + eps) /
                         (np.sum((est - tgt) ** 2) + eps))


def stoi_of(ref, est, sr=SR):
    try:
        from pystoi import stoi
        n = min(len(ref), len(est))
        return stoi(ref[:n].astype(np.float64), est[:n].astype(np.float64),
                    sr, extended=False)
    except Exception:
        return None


def spec_db(x, ref, n_fft=1024, hop=256):
    """Magnitude spectrogram in dB against a SHARED reference level."""
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop))
    return librosa.amplitude_to_db(S, ref=ref)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", required=True)
    p.add_argument("--noise", required=True)
    p.add_argument("--snr", type=float, default=0.0)
    p.add_argument("--ckpt")
    p.add_argument("--skip-dfn", action="store_true")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--fmax", type=int, default=8000)
    p.add_argument("--out", default="../results/fig_spectrograms.png")
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--mask-floor", type=float, default=0.15)
    p.add_argument("--play", action="store_true",
                   help="play noisy then enhanced, level-matched, so one "
                        "screen recording captures both")
    p.add_argument("--save-audio", action="store_true",
                   help="also write the wavs so you can play them in the demo")
    args = p.parse_args()

    seg = int(args.seconds * SR)
    speech, _ = librosa.load(args.clean, sr=SR, mono=True)
    speech = (np.pad(speech, (0, max(0, seg - len(speech))))[:seg]).astype(np.float32)
    noise, _ = librosa.load(args.noise, sr=SR, mono=True)

    print("\n" + "=" * 62)
    print("  STEP 1  load two files never seen in training")
    print("=" * 62)
    print(f"  speech : {os.path.basename(args.clean)}")
    print(f"  noise  : {os.path.basename(args.noise)}")

    ps = float(np.mean(speech ** 2))
    pn = float(np.mean(noise[:len(speech)] ** 2)) + 1e-12
    alpha = np.sqrt(ps / (pn * (10 ** (args.snr / 10))))
    print(f"\n  STEP 2  mix on the fly at {args.snr:+.0f} dB SNR")
    print(f"          speech power {ps:.6f}   noise power {pn:.6f}")
    print(f"          scale noise by alpha = sqrt(Ps / (Pn * 10^(SNR/10)))"
          f" = {alpha:.4f}")
    print("          the SPEECH is never scaled, so the reference is exact")

    noisy, clean = mix_at_snr(speech, noise.astype(np.float32), args.snr)
    print(f"\n  STEP 3  run the model, streaming, "
          f"{'mask floor ' + str(args.mask_floor) if args.mask_floor else 'no floor'}")

    panels = [("Clean reference", clean), ("Noisy input", noisy)]

    if not args.skip_dfn:
        try:
            from df.enhance import enhance, init_df
            m, st, _ = init_df()
            out = enhance(m, st, torch.from_numpy(noisy).unsqueeze(0))
            panels.append(("DeepFilterNet3", out.squeeze(0).numpy()))
        except Exception as e:
            print(f"DeepFilterNet unavailable ({e}) - skipping panel")

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        saved = ck.get("args", {})
        net = MaskNet(hidden=saved.get("hidden", 256),
                      layers=saved.get("layers", 2))
        net.load_state_dict(ck["model"])

        # Streaming path, deployment defaults - the same code that runs
        # live. Output lags the input by n_fft - hop, so we trim that
        # delay off to keep the panels time-aligned with the others.
        chain = EnhancementChain(net, 1024, args.hop,
                                 mask_floor=args.mask_floor,
                                 snr_aware=False, lms=False)
        for _ in range(20):
            chain.process_block(np.zeros(args.hop, dtype=np.float32))
        chain.reset()
        est = chain.process_signal(noisy.astype(np.float32))
        d = chain.latency_samples
        est = np.concatenate([est[d:], np.zeros(d, dtype=np.float32)])
        panels.append((f"Ours ({ck.get('epoch','?')} ep, streaming)", est))

    # ONE reference level for every panel - peak of the clean signal.
    ref = np.max(np.abs(librosa.stft(clean, n_fft=1024, hop_length=256)))

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(4.6 * len(panels), 4.4), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    img = None
    for ax, (title, sig) in zip(axes, panels):
        D = spec_db(sig.astype(np.float32), ref)
        img = librosa.display.specshow(D, sr=SR, hop_length=256,
                                       x_axis="time", y_axis="hz",
                                       ax=ax, vmin=-80, vmax=0)
        ax.set_title(title)
        ax.set_ylim(0, args.fmax)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (Hz)")

    fig.suptitle(f"{os.path.basename(args.noise)} at {args.snr:.0f} dB SNR "
                 f"- shared dB scale", y=1.02)
    cb = fig.colorbar(img, ax=axes, format="%+2.0f dB", pad=0.015)
    cb.set_label("level relative to clean peak")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")

    if args.save_audio:
        base = os.path.splitext(args.out)[0]
        for title, sig in panels:
            tag = title.split()[0].lower().strip("(),")
            path = f"{base}_{tag}.wav"
            sf.write(path, sig.astype(np.float32), SR)
            print(f"wrote {path}")

    # ---- measured results -------------------------------------------
    est_sig = None
    for title, sig in panels:
        if title.startswith("Ours"):
            est_sig = sig
    if est_sig is not None:
        print("\n" + "=" * 62)
        print("  STEP 4  measure")
        print("=" * 62)
        print(f"  {'':18}{'SI-SNR':>10}{'STOI':>9}")
        print("  " + "-" * 37)
        rows = []
        for name, sig in [("noisy input", noisy), ("our output", est_sig)]:
            sv = si_snr(clean, sig)
            tv = stoi_of(clean, sig)
            rows.append((sv, tv))
            print(f"  {name:18}{sv:>10.2f}{(f'{tv:.3f}' if tv else 'n/a'):>9}")
        print("  " + "-" * 37)
        d_si = rows[1][0] - rows[0][0]
        line = f"  {'improvement':18}{d_si:>+10.2f}"
        if rows[0][1] and rows[1][1]:
            line += f"{rows[1][1] - rows[0][1]:>+9.3f}"
        print(line)

    if args.play:
        try:
            import sounddevice as sd
            import time as _t
            for name, sig in [("NOISY", noisy), ("OURS", est_sig)]:
                if sig is None:
                    continue
                x = sig / (np.max(np.abs(sig)) + 1e-9) * 0.9   # match level
                print(f"\n  playing {name} ...")
                sd.play(x.astype(np.float32), SR); sd.wait(); _t.sleep(0.6)
        except Exception as e:
            print(f"  playback unavailable: {e}")

    print(f"\nOur panel used the STREAMING path (hop {args.hop}, mask "
          f"floor {args.mask_floor}) - the same code that runs live,")
    print("with the 16 ms algorithmic delay removed for alignment.")
    print("\nTip: for the deck, pick a clip with a clear impulse in it.")
    print("An impulse shows as a bright VERTICAL stripe spanning all")
    print("frequencies - the judge can see it survive or vanish.")


if __name__ == "__main__":
    main()
