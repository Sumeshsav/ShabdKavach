"""
SIH baseline evidence generator.

Mixes clean speech with noise at controlled SNRs, runs DeepFilterNet,
scores the result, and saves before/after spectrograms for the deck.

WHAT YOU NEED
  clean.wav  - a clean speech recording (LibriSpeech / VCTK / your own voice)
  noise.wav  - a noise recording (UrbanSound8K gunshot, DEMAND, helicopter...)
Both are resampled to 48 kHz automatically.

RUN
  python baseline.py clean.wav noise.wav

OUTPUT
  results/  - mixed + denoised wav files, spectrogram PNGs
  a printed table of STOI / SI-SNR (and PESQ if installed)
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")           # no GUI needed
import matplotlib.pyplot as plt

SR = 48000                      # DeepFilterNet expects 48 kHz
SNR_LEVELS = [0, 5, 10]         # dB - report across a range, never one number
OUT = "results"


# ----------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------
# SI-SNR: scale-invariant signal-to-noise ratio, in dB.
# "Scale-invariant" means making the whole signal louder doesn't fake a
# better score - it projects the estimate onto the reference first.
def si_snr(reference, estimate, eps=1e-8):
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    # project estimate onto reference -> the part that is real signal
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    target = alpha * reference
    noise = estimate - target
    return 10 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps))


def score(reference, estimate, sr=SR):
    """Returns dict of metrics. PESQ is optional (no Windows wheel)."""
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]
    out = {"si_snr": si_snr(reference, estimate)}

    try:
        from pystoi import stoi
        out["stoi"] = stoi(reference, estimate, sr, extended=False)
    except Exception as e:
        out["stoi"] = None
        print(f"  [stoi unavailable: {e}]")

    try:
        from pesq import pesq
        # PESQ only accepts 8k (narrowband) or 16k (wideband)
        r16 = librosa.resample(reference, orig_sr=sr, target_sr=16000)
        e16 = librosa.resample(estimate, orig_sr=sr, target_sr=16000)
        out["pesq"] = pesq(16000, r16, e16, "wb")
    except Exception:
        out["pesq"] = None      # expected on Windows - not an error
    return out


# ----------------------------------------------------------------------
# MIXING
# ----------------------------------------------------------------------
# To hit a target SNR we scale the NOISE, never the speech - so the
# reference signal stays byte-identical across all SNR levels and the
# metrics stay comparable.
def mix_at_snr(speech, noise, snr_db):
    # tile or trim noise to match speech length
    if len(noise) < len(speech):
        reps = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:len(speech)]

    p_speech = np.mean(speech ** 2)
    p_noise = np.mean(noise ** 2) + 1e-12
    # solve: SNR = 10*log10(P_speech / (scale^2 * P_noise))
    scale = np.sqrt(p_speech / (p_noise * (10 ** (snr_db / 10))))
    mixed = speech + scale * noise

    peak = np.max(np.abs(mixed))
    if peak > 0.99:                       # avoid clipping
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


# ----------------------------------------------------------------------
# PLOTTING - this is what goes in the deck
# ----------------------------------------------------------------------
def spectrogram_pair(noisy, clean_out, sr, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, sig, name in [(axes[0], noisy, "Noisy input"),
                          (axes[1], clean_out, "After denoising")]:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=1024,
                                                        hop_length=256)),
                                    ref=np.max)
        img = librosa.display.specshow(D, sr=sr, hop_length=256,
                                       x_axis="time", y_axis="hz", ax=ax)
        ax.set_title(name)
        ax.set_ylim(0, 8000)          # speech lives below 8 kHz
    fig.suptitle(title)
    fig.colorbar(img, ax=axes, format="%+2.0f dB")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clean", help="clean speech wav")
    ap.add_argument("noise", help="noise wav")
    ap.add_argument("--snrs", type=int, nargs="+", default=SNR_LEVELS)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    print("Loading audio...")
    speech, _ = librosa.load(args.clean, sr=SR, mono=True)
    noise, _ = librosa.load(args.noise, sr=SR, mono=True)
    print(f"  speech {len(speech)/SR:.1f}s   noise {len(noise)/SR:.1f}s")

    print("Loading DeepFilterNet (first run downloads the model)...")
    from df.enhance import enhance, init_df
    import torch
    model, df_state, _ = init_df()

    rows = []
    for snr in args.snrs:
        print(f"\n--- {snr} dB SNR ---")
        noisy = mix_at_snr(speech, noise, snr)
        sf.write(f"{OUT}/noisy_{snr}dB.wav", noisy, SR)

        t0 = time.time()
        enhanced = enhance(model, df_state,
                           torch.from_numpy(noisy).unsqueeze(0))
        elapsed = time.time() - t0
        enhanced = enhanced.squeeze(0).numpy()
        sf.write(f"{OUT}/denoised_{snr}dB.wav", enhanced, SR)

        # real-time factor: <1.0 means faster than real time
        rtf = elapsed / (len(noisy) / SR)

        before = score(speech, noisy)
        after = score(speech, enhanced)
        rows.append((snr, before, after, rtf))

        spectrogram_pair(noisy, enhanced, SR,
                         f"{snr} dB SNR input",
                         f"{OUT}/spectrogram_{snr}dB.png")

    # ---- results table: paste this straight into the deck ----
    print("\n" + "=" * 74)
    print(f"{'SNR in':>7} {'STOI before':>12} {'STOI after':>11} "
          f"{'SI-SNR before':>14} {'SI-SNR after':>13} {'RTF':>6}")
    print("-" * 74)
    for snr, b, a, rtf in rows:
        sb = f"{b['stoi']:.3f}" if b["stoi"] is not None else "n/a"
        sa = f"{a['stoi']:.3f}" if a["stoi"] is not None else "n/a"
        print(f"{snr:>5} dB {sb:>12} {sa:>11} "
              f"{b['si_snr']:>13.2f} {a['si_snr']:>12.2f} {rtf:>6.2f}")
    if any(r[2]["pesq"] is not None for r in rows):
        print("\nPESQ (wideband):")
        for snr, b, a, _ in rows:
            print(f"  {snr:>2} dB: {b['pesq']:.2f} -> {a['pesq']:.2f}")
    else:
        print("\nPESQ unavailable (no Windows wheel) - run in Colab for PESQ.")
    print("=" * 74)
    print(f"\nRTF below 1.0 = faster than real time. Files in {OUT}/")


if __name__ == "__main__":
    main()
