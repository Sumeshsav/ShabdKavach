"""
make_split.py - carve a held-out test set out of your data folders.

WHY THIS MATTERS
If you evaluate on files the model trained on, the score is inflated and
an evaluator can ask about it. Held-out means: moved aside BEFORE
training and never seen by the model.

WHAT IT DOES
  - moves N speech files to data/speech_test/
  - moves K noise files PER CATEGORY to data/noise_test/
Files are MOVED, not copied, so they cannot leak back into training.

Selection is seeded, so the split is reproducible - you can state the
seed in your report and anyone can recreate it.

USAGE
  python make_split.py --dry-run          # see what would move
  python make_split.py                    # actually move
  python make_split.py --n-speech 40 --n-noise-per-cat 2

AFTER RUNNING: retrain. The old checkpoint saw the test files.
"""

import argparse
import glob
import os
import random
import shutil
from collections import defaultdict


def category_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0] if "_" in stem else "uncategorised"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--speech", default="../data/speech")
    p.add_argument("--noise", default="../data/noise")
    p.add_argument("--speech-test", default="../data/speech_test")
    p.add_argument("--noise-test", default="../data/noise_test")
    p.add_argument("--n-speech", type=int, default=30)
    p.add_argument("--n-noise-per-cat", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rng = random.Random(args.seed)
    speech = sorted(glob.glob(os.path.join(args.speech, "*.wav")))
    noise = sorted(glob.glob(os.path.join(args.noise, "*.wav")))

    if not speech or not noise:
        print(f"Found {len(speech)} speech and {len(noise)} noise files. "
              "Fill both folders first.")
        return

    # ---- speech: random subset ----
    n = min(args.n_speech, max(0, len(speech) - 1))
    speech_pick = rng.sample(speech, n)

    # ---- noise: K per category, so every category is represented in
    #      BOTH train and test. Skip categories with only one file -
    #      moving it would leave training with none of that type.
    by_cat = defaultdict(list)
    for f in noise:
        by_cat[category_of(f)].append(f)

    noise_pick, skipped = [], []
    for cat, files in sorted(by_cat.items()):
        if len(files) <= args.n_noise_per_cat:
            skipped.append((cat, len(files)))
            continue
        noise_pick += rng.sample(files, args.n_noise_per_cat)

    print(f"speech: {len(speech)} total -> moving {len(speech_pick)} to test")
    print(f"noise : {len(noise)} total -> moving {len(noise_pick)} to test")
    for cat, k in sorted(by_cat.items()):
        mark = "  SKIPPED (too few files)" if any(c == cat for c, _ in skipped) else ""
        print(f"    {cat:<16} {len(k):>4} files{mark}")

    if skipped:
        print("\nCategories were skipped because moving their only file would")
        print("leave the training set with none of that noise type. Collect")
        print("more of those before splitting them.")

    if args.dry_run:
        print("\n--dry-run: nothing moved. Re-run without it to apply.")
        return

    os.makedirs(args.speech_test, exist_ok=True)
    os.makedirs(args.noise_test, exist_ok=True)
    for f in speech_pick:
        shutil.move(f, os.path.join(args.speech_test, os.path.basename(f)))
    for f in noise_pick:
        shutil.move(f, os.path.join(args.noise_test, os.path.basename(f)))

    print(f"\nmoved. train now has "
          f"{len(glob.glob(os.path.join(args.speech,'*.wav')))} speech, "
          f"{len(glob.glob(os.path.join(args.noise,'*.wav')))} noise")
    print(f"test  now has "
          f"{len(glob.glob(os.path.join(args.speech_test,'*.wav')))} speech, "
          f"{len(glob.glob(os.path.join(args.noise_test,'*.wav')))} noise")
    print(f"\nsplit seed = {args.seed} (state this in your report)")
    print("NOW RETRAIN - your old checkpoint saw these test files.")


if __name__ == "__main__":
    main()
