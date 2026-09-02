"""
extract_urbansound.py - pull specific labelled classes out of UrbanSound8K.

UrbanSound8K ships as 10 numbered "fold" folders plus a metadata CSV.
The gunshot clips you need are scattered across all 10 folds. This reads
the CSV, finds the classes you want, and copies them into one folder with
the category prefix dataset.py expects.

CLASS -> CATEGORY MAPPING (edit MAPPING below if you disagree)
  gun_shot, dog_bark, car_horn            -> impulse
  air_conditioner, engine_idling,
  jackhammer, drilling                    -> steady
  siren                                   -> nonstat
  children_playing, street_music          -> babble

USAGE
  python extract_urbansound.py C:\\path\\to\\UrbanSound8K ..\\data\\raw_noise
  python extract_urbansound.py <root> <out> --classes gun_shot siren
  python extract_urbansound.py <root> <out> --max-per-class 60

Then convert to 48 kHz with prep_data.py (see the printed hint).
"""

import argparse
import csv
import os
import shutil

MAPPING = {
    "gun_shot": "impulse",
    "dog_bark": "impulse",
    "car_horn": "impulse",
    "air_conditioner": "steady",
    "engine_idling": "steady",
    "jackhammer": "steady",
    "drilling": "steady",
    "siren": "nonstat",
    "children_playing": "babble",
    "street_music": "babble",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="UrbanSound8K folder (contains audio/ and metadata/)")
    p.add_argument("out", help="where to copy the selected clips")
    p.add_argument("--classes", nargs="+", default=list(MAPPING.keys()))
    p.add_argument("--max-per-class", type=int, default=80,
                   help="cap per class so one class does not dominate")
    p.add_argument("--min-duration", type=float, default=0.5,
                   help="skip clips shorter than this (seconds)")
    args = p.parse_args()

    meta = os.path.join(args.root, "metadata", "UrbanSound8K.csv")
    if not os.path.exists(meta):
        print(f"Could not find {meta}")
        print("Point 'root' at the folder that contains audio/ and metadata/")
        return

    os.makedirs(args.out, exist_ok=True)
    counts = {}
    copied = 0
    missing = 0

    with open(meta, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = row["class"]
            if cls not in args.classes:
                continue
            if counts.get(cls, 0) >= args.max_per_class:
                continue
            try:
                dur = float(row["end"]) - float(row["start"])
            except (KeyError, ValueError):
                dur = 4.0
            if dur < args.min_duration:
                continue

            src = os.path.join(args.root, "audio",
                               f"fold{row['fold']}", row["slice_file_name"])
            if not os.path.exists(src):
                missing += 1
                continue

            category = MAPPING.get(cls, "uncategorised")
            stem = os.path.splitext(row["slice_file_name"])[0]
            dst = os.path.join(args.out, f"{category}_{cls}-{stem}.wav")
            shutil.copy2(src, dst)
            counts[cls] = counts.get(cls, 0) + 1
            copied += 1

    print(f"copied {copied} clips to {args.out}/")
    if missing:
        print(f"({missing} listed files were not found on disk)")
    for cls in sorted(counts):
        print(f"  {cls:<20} {counts[cls]:>4}  -> {MAPPING.get(cls)}")

    print("\nThese are 44.1 kHz. Convert them to 48 kHz mono next:")
    print(f"  python prep_data.py {args.out} ..\\data\\noise --split 10")
    print("(prep_data.py keeps the category prefix already in the filename)")


if __name__ == "__main__":
    main()
