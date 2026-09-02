"""
compare.py - fair head-to-head comparison for the PPT.

WHY THIS FILE EXISTS
You cannot compare a validation SI-SNR averaged over random SNRs
against a baseline measured at fixed 0/5/10 dB on different files.
Different test data and different SNR distributions make the numbers
meaningless side by side.

This script runs EVERY system on the SAME held-out files at the SAME
fixed SNRs, with the same seed, and prints one table.

HELD-OUT DATA IS MANDATORY
Your model trained on data/speech and data/noise. Evaluating on those
is evaluating on the training set - the numbers would be inflated and
an evaluator can ask about it. Make a proper split first:

    mkdir ..\\data\\speech_test  ..\\data\\noise_test
    # move ~30 speech files and 1 noise file per category across,
    # then RETRAIN on what remains.

USAGE
  python compare.py --speech-test ../data/speech_test \\
                    --noise-test  ../data/noise_test \\
                    --ckpt checkpoints/best.pt

  # skip DeepFilterNet if it is not installed on this machine
  python compare.py ... --skip-dfn
"""

import argparse
import csv
import glob
import os
import random
import time
from collections import defaultdict

import numpy as np
import librosa
import torch

from dataset import mix_at_snr, category_of
from model import MaskNet

SR = 48000


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def si_snr_np(reference, estimate, eps=1e-8):
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    target = alpha * reference
    noise = estimate - target
    return 10 * np.log10((np.sum(target ** 2) + eps) /
                         (np.sum(noise ** 2) + eps))


def score(reference, estimate, sr=SR):
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]
    out = {"si_snr": si_snr_np(reference, estimate)}
    try:
        from pystoi import stoi
        out["stoi"] = stoi(reference, estimate, sr, extended=False)
    except Exception:
        out["stoi"] = None
    try:
        from pesq import pesq
        r = librosa.resample(reference, orig_sr=sr, target_sr=16000)
        e = librosa.resample(estimate, orig_sr=sr, target_sr=16000)
        out["pesq"] = pesq(16000, r, e, "wb")
    except Exception:
        out["pesq"] = None
    return out


# ----------------------------------------------------------------------
# systems under test
# ----------------------------------------------------------------------
class Unprocessed:
    """The noisy input itself - the floor every system must beat."""
    name = "Unprocessed (noisy)"
    params = 0
    trained_on = "-"

    def __call__(self, noisy):
        return noisy


class OurModel:
    def __init__(self, ckpt_path, trained_on="40 min speech, 4 noise files"):
        ck = torch.load(ckpt_path, map_location="cpu")
        saved = ck.get("args", {})
        self.model = MaskNet(hidden=saved.get("hidden", 256),
                             layers=saved.get("layers", 2))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.epoch = ck.get("epoch", "?")
        self.params = self.model.count_parameters()
        self.trained_on = trained_on
        self.name = f"Ours (MaskNet, {self.epoch} epochs)"

    @torch.no_grad()
    def __call__(self, noisy):
        x = torch.from_numpy(noisy).unsqueeze(0)
        return self.model(x).squeeze(0).numpy()


class DeepFilterNet:
    name = "DeepFilterNet3 (pretrained)"
    params = None                      # filled after load
    trained_on = "DNS Challenge, 100s of hours"

    def __init__(self):
        from df.enhance import enhance, init_df
        self._enhance = enhance
        self.model, self.state, _ = init_df()
        try:
            self.params = sum(p.numel() for p in self.model.parameters())
        except Exception:
            self.params = None

    def __call__(self, noisy):
        out = self._enhance(self.model, self.state,
                            torch.from_numpy(noisy).unsqueeze(0))
        return out.squeeze(0).numpy()


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--speech-test", required=True)
    p.add_argument("--noise-test", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--snrs", type=int, nargs="+", default=[0, 5, 10])
    p.add_argument("--pairs", type=int, default=20,
                   help="how many (speech, noise) pairs per SNR")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--skip-dfn", action="store_true")
    p.add_argument("--out", default="../results/comparison.csv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    speech_files = sorted(glob.glob(os.path.join(args.speech_test, "*.wav")))
    noise_files = sorted(glob.glob(os.path.join(args.noise_test, "*.wav")))
    if not speech_files or not noise_files:
        print("Need wav files in BOTH --speech-test and --noise-test.")
        return
    print(f"held-out: {len(speech_files)} speech, {len(noise_files)} noise")

    # ---- build the systems ----
    systems = [Unprocessed()]
    if not args.skip_dfn:
        try:
            systems.append(DeepFilterNet())
        except Exception as e:
            print(f"DeepFilterNet unavailable ({e}) - skipping")
    if args.ckpt:
        systems.append(OurModel(args.ckpt))

    # ---- fixed pair list, identical for every system ----
    rng = random.Random(args.seed)
    seg = int(args.seconds * SR)
    pairs = []
    for _ in range(args.pairs):
        sp = rng.choice(speech_files)
        no = rng.choice(noise_files)
        speech, _ = librosa.load(sp, sr=SR, mono=True)
        if len(speech) <= seg:
            speech = np.pad(speech, (0, seg - len(speech)))
        else:
            st = rng.randint(0, len(speech) - seg)
            speech = speech[st:st + seg]
        noise, _ = librosa.load(no, sr=SR, mono=True)
        pairs.append((speech.astype(np.float32), noise.astype(np.float32),
                      category_of(no)))

    rows = []
    for snr in args.snrs:
        mixtures = [(mix_at_snr(s, n, snr), c) for s, n, c in pairs]
        for sysm in systems:
            print(f"  {sysm.name} @ {snr} dB ...")
            for (noisy, clean), cat in mixtures:
                t0 = time.perf_counter()
                est = sysm(noisy)
                rtf = (time.perf_counter() - t0) / (len(noisy) / SR)
                m = score(clean.astype(np.float64), est.astype(np.float64))
                rows.append({"system": sysm.name, "snr_db": snr,
                             "category": cat, "rtf": rtf, **m})

    # ---- write raw rows ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")

    # ---- main table ----
    def agg(key, sysname, snr, cat=None):
        vals = [r[key] for r in rows
                if r["system"] == sysname and r["snr_db"] == snr
                and r[key] is not None and (cat is None or r["category"] == cat)]
        return np.mean(vals) if vals else None

    def fmt(v, w=8, p=3):
        return f"{v:{w}.{p}f}" if v is not None else " " * (w - 3) + "n/a"

    print("\n" + "=" * 88)
    print("HEAD TO HEAD - identical held-out files, identical SNRs, same seed")
    print("=" * 88)
    print(f"{'system':<32}{'SNR':>5}{'STOI':>9}{'SI-SNR':>10}"
          f"{'PESQ':>9}{'RTF':>8}")
    print("-" * 88)
    for snr in args.snrs:
        for sysm in systems:
            print(f"{sysm.name:<32}{snr:>4}dB"
                  f"{fmt(agg('stoi', sysm.name, snr), 9)}"
                  f"{fmt(agg('si_snr', sysm.name, snr), 10, 2)}"
                  f"{fmt(agg('pesq', sysm.name, snr), 9, 2)}"
                  f"{fmt(agg('rtf', sysm.name, snr), 8)}")
        print("-" * 88)

    # ---- context table: this is what makes the comparison honest ----
    print("\nCONTEXT (quote this next to the results)")
    print(f"{'system':<32}{'parameters':>14}{'trained on':>34}")
    print("-" * 80)
    for sysm in systems:
        pr = f"{sysm.params:,}" if sysm.params else "-"
        print(f"{sysm.name:<32}{pr:>14}{sysm.trained_on:>34}")

    # ---- per-category, at the hardest SNR ----
    cats = sorted({r["category"] for r in rows})
    if len(cats) > 1:
        snr0 = args.snrs[0]
        print(f"\nBY NOISE CATEGORY at {snr0} dB (STOI)")
        print(f"{'system':<32}" + "".join(f"{c:>12}" for c in cats))
        print("-" * (32 + 12 * len(cats)))
        for sysm in systems:
            line = f"{sysm.name:<32}"
            for c in cats:
                line += fmt(agg("stoi", sysm.name, snr0, c), 12)
            print(line)

    print("\nHonest reading: report BOTH tables together. A smaller model")
    print("trained on far less data scoring lower is expected, not a")
    print("failure - the parameter and training-data columns are what make")
    print("the comparison fair.")


if __name__ == "__main__":
    main()
