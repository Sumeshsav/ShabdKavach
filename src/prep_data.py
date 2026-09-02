"""
prep_data.py - turn a messy download into the flat 48 kHz wav folder
that dataset.py expects.

WHY THIS EXISTS
LibriSpeech ships as nested folders of .flac at 16 kHz:
    dev-clean/1272/128104/1272-128104-0000.flac
Freesound gives you .mp3/.ogg/.wav at assorted sample rates.
dataset.py wants:
    data/speech/*.wav   all 48 kHz mono

USAGE
  # speech: walk the whole LibriSpeech tree, cap at 40 minutes
  python prep_data.py dev-clean data/speech --max-minutes 40

  # noise: convert downloads and tag them with a category prefix
  python prep_data.py downloads/gunshots data/noise --prefix impulse
  python prep_data.py downloads/engines  data/noise --prefix steady

  # split long recordings into 10-second chunks
  python prep_data.py downloads/ambience data/noise --prefix steady --split 10
"""

import argparse
import os

import numpy as np
import librosa
import soundfile as sf

SR = 48000
EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


def find_audio(root):
    out = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in EXTS:
                out.append(os.path.join(dirpath, f))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", help="folder to search (searched recursively)")
    p.add_argument("dst", help="output folder")
    p.add_argument("--prefix", default="",
                   help="category prefix for noise, e.g. impulse / steady")
    p.add_argument("--max-minutes", type=float, default=None,
                   help="stop once this much audio has been written")
    p.add_argument("--split", type=float, default=None,
                   help="chop long files into chunks of this many seconds")
    p.add_argument("--min-seconds", type=float, default=1.0,
                   help="skip clips shorter than this")
    p.add_argument("--sr", type=int, default=SR)
    args = p.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = find_audio(args.src)
    if not files:
        print(f"No audio found under {args.src}")
        return
    print(f"Found {len(files)} audio files")

    written = 0
    total_seconds = 0.0
    budget = args.max_minutes * 60 if args.max_minutes else None

    for path in files:
        if budget and total_seconds >= budget:
            print(f"Reached {args.max_minutes} minute budget - stopping.")
            break
        try:
            audio, _ = librosa.load(path, sr=args.sr, mono=True)
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {e}")
            continue

        if len(audio) < args.min_seconds * args.sr:
            continue

        # Normalise peak to 0.9. Downloads vary wildly in level, and
        # mix_at_snr sets the SNR anyway - but consistent levels stop
        # one very quiet file from being effectively ignored.
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.9 / peak)

        chunks = [audio]
        if args.split:
            n = int(args.split * args.sr)
            chunks = [audio[i:i + n] for i in range(0, len(audio), n)]
            chunks = [c for c in chunks if len(c) >= args.min_seconds * args.sr]

        stem = os.path.splitext(os.path.basename(path))[0]
        stem = "".join(ch if ch.isalnum() else "-" for ch in stem)[:40]

        for k, chunk in enumerate(chunks):
            if budget and total_seconds >= budget:
                break
            suffix = f"-{k:03d}" if len(chunks) > 1 else ""
            name = f"{args.prefix}_{stem}{suffix}.wav" if args.prefix \
                   else f"{stem}{suffix}.wav"
            sf.write(os.path.join(args.dst, name),
                     chunk.astype(np.float32), args.sr)
            written += 1
            total_seconds += len(chunk) / args.sr

    print(f"\nWrote {written} files, {total_seconds/60:.1f} minutes "
          f"to {args.dst}/ at {args.sr} Hz mono")
    if args.prefix:
        print(f"All named '{args.prefix}_*' so dataset.py groups them "
              f"as category '{args.prefix}'")


if __name__ == "__main__":
    main()
