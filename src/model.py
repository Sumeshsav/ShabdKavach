"""
model.py - a small causal mask-estimation network.

WHAT IT DOES, IN ONE LINE
Look at the noisy spectrogram, predict a number between 0 and 1 for
every time-frequency cell, and multiply the spectrogram by those
numbers. Cells that are mostly speech get ~1 (keep), cells that are
mostly noise get ~0 (remove).

WHY A MASK INSTEAD OF GENERATING CLEAN AUDIO
Generating audio from scratch is hard. Deciding "keep or discard" for
each cell is much easier to learn, and it can never invent sounds that
were not in the input - so it cannot hallucinate speech.

WHY GRU AND NOT SOMETHING FANCIER
A GRU reads frames in order and only ever sees the PAST. That is what
"causal" means, and causal is what makes real-time possible. A
bidirectional layer or full attention would need the whole clip up
front - great offline metrics, useless in a radio.
"""

import torch
import torch.nn as nn


class MaskNet(nn.Module):
    def __init__(self, n_fft=1024, hop=256, hidden=256, layers=2, sr=48000):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.sr = sr
        self.n_bins = n_fft // 2 + 1          # 513 for n_fft=1024

        # Hann window, tapered at the edges to avoid spectral leakage.
        # register_buffer so it moves to GPU with the model but is not
        # a trainable parameter.
        self.register_buffer("window", torch.hann_window(n_fft))

        self.gru = nn.GRU(
            input_size=self.n_bins,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=False,              # <- causality lives here
        )
        self.out = nn.Linear(hidden, self.n_bins)

    # ---- spectral helpers -------------------------------------------
    def stft(self, x):
        """[B, samples] -> complex [B, F, T]"""
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                          window=self.window, return_complex=True)

    def istft(self, spec, length):
        """complex [B, F, T] -> [B, samples]"""
        return torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop,
                           window=self.window, length=length)

    # ---- forward -----------------------------------------------------
    def forward(self, noisy, return_mask=False):
        length = noisy.shape[-1]
        spec = self.stft(noisy)                       # [B, F, T] complex
        mag = spec.abs()                              # magnitude only

        # log1p compresses the huge dynamic range of audio so the
        # network sees quiet detail as well as loud peaks. log1p(x) =
        # log(1+x), which is safe at x=0 unlike plain log.
        feat = torch.log1p(mag).transpose(1, 2)       # [B, T, F]

        h, _ = self.gru(feat)                         # [B, T, hidden]
        mask = torch.sigmoid(self.out(h))             # [B, T, F] in 0..1
        mask = mask.transpose(1, 2)                   # [B, F, T]

        # Apply the mask to the COMPLEX spectrum. Multiplying a complex
        # number by a real 0..1 scales its magnitude and leaves phase
        # untouched - we reuse the noisy phase. That is the standard
        # simplification; predicting phase too is the next upgrade.
        enhanced = self.istft(spec * mask, length)

        if return_mask:
            return enhanced, mask
        return enhanced

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = MaskNet()
    x = torch.randn(2, 48000)                         # 2 clips of 1 second
    y = m(x)
    print("in ", tuple(x.shape))
    print("out", tuple(y.shape))
    print(f"parameters: {m.count_parameters():,}")
