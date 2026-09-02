"""
dataset.py - turns your speech/ and noise/ folders into training pairs.

THE IDEA
A neural network learns from EXAMPLES: "here is a noisy clip, here is
what you should have produced". We never store those pairs on disk -
we build them fresh every time one is requested. That means every pass
over the data sees new speech+noise+SNR combinations, which is free
data augmentation and stops the model memorising specific clips.

FOLDER LAYOUT
    data/speech/*.wav          clean speech, any length
    data/noise/steady_*.wav    noise, NAMED BY CATEGORY
    data/noise/impulse_*.wav
    ...

The category prefix (text before the first underscore) is returned with
each item so you can report results per noise type later.
"""

import glob
import os
import random

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

SR = 48000


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float):
    """Scale the NOISE so the mixture has the requested SNR.

    Power of a signal is its mean square:      P = mean(x^2)
    SNR in dB is:                              SNR = 10*log10(Ps/Pn)
    We scale noise by alpha, and scaling a signal by alpha scales its
    POWER by alpha^2, so:
        SNR_target = 10*log10( Ps / (alpha^2 * Pn) )
    Solving for alpha:
        alpha = sqrt( Ps / (Pn * 10^(SNR/10)) )

    We scale the noise and never the speech, so the clean reference is
    identical at every SNR and the metrics stay comparable.
    """
    if len(noise) < len(speech):                      # tile short noise
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    noise = noise[:len(speech)]

    p_speech = np.mean(speech ** 2)
    p_noise = np.mean(noise ** 2) + 1e-12
    alpha = np.sqrt(p_speech / (p_noise * (10 ** (snr_db / 10))))
    mixed = speech + alpha * noise

    # WAV floats live in -1..+1. Past that the waveform clips and sounds
    # broken. Scaling the MIXTURE scales speech and noise equally, so
    # the SNR we just set is preserved.
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
        speech = speech * (0.99 / peak)               # keep the pair aligned
    return mixed.astype(np.float32), speech.astype(np.float32)


def category_of(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0] if "_" in stem else "uncategorised"


class NoisySpeechDataset(Dataset):
    """Yields (noisy, clean) pairs of `segment_seconds` each.

    Args:
        speech_dir, noise_dir: folders of wav files
        snr_range: mixtures are drawn uniformly from this range, in dB.
            Train on a RANGE, not one value, or the model only learns
            one noise level.
        segment_seconds: short crops keep GPU memory small and let you
            fit more examples in a batch.
        length: how many pairs count as one "epoch". Because mixing is
            random, this is a free choice rather than a fixed dataset size.
    """

    def __init__(self, speech_dir, noise_dir, snr_range=(-5, 15),
                 segment_seconds=4.0, sr=SR, length=2000, seed=None):
        self.speech_files = sorted(glob.glob(os.path.join(speech_dir, "*.wav")))
        self.noise_files = sorted(glob.glob(os.path.join(noise_dir, "*.wav")))
        if not self.speech_files:
            raise FileNotFoundError(f"no wav files in {speech_dir}")
        if not self.noise_files:
            raise FileNotFoundError(f"no wav files in {noise_dir}")

        self.sr = sr
        self.seg = int(segment_seconds * sr)
        self.snr_range = snr_range
        self.length = length
        self.rng = random.Random(seed)
        self._cache = {}          # decoded audio, loaded once per file

    def __len__(self):
        return self.length

    # ---- helpers ----------------------------------------------------
    def _load(self, path):
        """Decode + resample once, then keep in memory."""
        if path not in self._cache:
            audio, _ = librosa.load(path, sr=self.sr, mono=True)
            self._cache[path] = audio.astype(np.float32)
        return self._cache[path]

    def _random_crop(self, audio):
        """Take a random `seg`-long window; pad with zeros if too short."""
        if len(audio) <= self.seg:
            return np.pad(audio, (0, self.seg - len(audio)))
        start = self.rng.randint(0, len(audio) - self.seg)
        return audio[start:start + self.seg]

    # ---- the actual item --------------------------------------------
    def __getitem__(self, idx):
        speech_path = self.rng.choice(self.speech_files)
        noise_path = self.rng.choice(self.noise_files)

        speech = self._random_crop(self._load(speech_path))

        # Skip near-silent crops - a pair with no speech teaches nothing
        # and makes SI-SNR explode. Try a few times, then accept.
        tries = 0
        while np.sqrt(np.mean(speech ** 2)) < 1e-4 and tries < 5:
            speech = self._random_crop(self._load(self.rng.choice(self.speech_files)))
            tries += 1

        noise = self._random_crop(self._load(noise_path))
        snr = self.rng.uniform(*self.snr_range)
        noisy, clean = mix_at_snr(speech, noise, snr)

        return {
            "noisy": torch.from_numpy(noisy),
            "clean": torch.from_numpy(clean),
            "snr_db": torch.tensor(snr, dtype=torch.float32),
            "category": category_of(noise_path),
        }


if __name__ == "__main__":
    # quick self-test: python src/dataset.py data/speech data/noise
    import sys
    ds = NoisySpeechDataset(sys.argv[1], sys.argv[2], length=8, seed=0)
    item = ds[0]
    print("noisy", item["noisy"].shape, "clean", item["clean"].shape,
          "snr", float(item["snr_db"]), "cat", item["category"])
    import soundfile as sf
    sf.write("check_noisy.wav", item["noisy"].numpy(), SR)
    sf.write("check_clean.wav", item["clean"].numpy(), SR)
    print("wrote check_noisy.wav / check_clean.wav - LISTEN to these before training")
