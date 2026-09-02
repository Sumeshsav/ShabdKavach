"""
losses.py - how wrong the model's guess was.

A loss is a single number the training loop tries to make SMALLER.
Everything the model learns comes from the shape of this number, so
the loss IS the specification of what "good" means.

THIS FILE IS WHERE YOUR INNOVATION LIVES.
The standard loss treats every moment of audio equally. A gunshot
occupies maybe 20 milliseconds out of 4 seconds - 0.5% of the frames.
Averaged over everything, the model can botch every gunshot and barely
be penalised, so it never learns to handle them.

impulse_weighted_loss() finds those frames and multiplies their
penalty, forcing the model to care about them.
"""

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# SI-SNR: how close is the output waveform to the clean reference?
# ----------------------------------------------------------------------
# Project the estimate onto the reference (the same vector projection
# from linear algebra), split it into "the part along the reference"
# and "everything else", and take their power ratio in dB.
#
#   alpha      = <est, ref> / <ref, ref>
#   s_target   = alpha * ref
#   e_noise    = est - s_target
#   SI-SNR     = 10*log10( ||s_target||^2 / ||e_noise||^2 )
#
# "Scale-invariant": multiply est by any constant k and alpha scales by
# k, so both terms scale by k^2 and cancel. You cannot improve the
# score by turning the volume up - only by actually removing noise.
def si_snr(estimate, reference, eps=1e-8):
    """[B, samples] -> [B] of dB values (higher is better)."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)

    dot = torch.sum(estimate * reference, dim=-1, keepdim=True)
    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + eps
    s_target = (dot / ref_energy) * reference
    e_noise = estimate - s_target

    ratio = (torch.sum(s_target ** 2, dim=-1) + eps) / \
            (torch.sum(e_noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio)


def si_snr_loss(estimate, reference):
    """Loss = NEGATIVE SI-SNR, because training minimises the loss but
    we want SI-SNR to go up."""
    return -si_snr(estimate, reference).mean()


# ----------------------------------------------------------------------
# Impulse detection
# ----------------------------------------------------------------------
def frame_energy_db(x, n_fft=1024, hop=256):
    """Energy of each short frame, in dB. [B, samples] -> [B, T]"""
    window = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window,
                      return_complex=True)
    energy = spec.abs().pow(2).sum(dim=1)              # [B, T]
    return 10 * torch.log10(energy + 1e-10)


def detect_impulse_frames(noisy, n_fft=1024, hop=256, threshold_db=8.0):
    """Flag frames whose energy spikes well above the clip's own median.

    A gunshot is a sudden broadband burst - far louder than the local
    average, and over in milliseconds. Comparing each frame to the
    MEDIAN (not the mean) makes the detector robust: a few very loud
    frames barely move the median, so they stand out clearly.

    Returns a float mask [B, T] of 1.0 (impulse) / 0.0 (normal).
    """
    e = frame_energy_db(noisy, n_fft, hop)             # [B, T]
    median = e.median(dim=-1, keepdim=True).values     # [B, 1]
    return (e > median + threshold_db).float()


# ----------------------------------------------------------------------
# Per-frame spectral error, so we can weight specific frames
# ----------------------------------------------------------------------
# SI-SNR is a single number for the whole clip - there is no way to say
# "this frame mattered more". So for the weighted part we use an L1
# error on the log-magnitude spectrogram, which IS per-frame.
def per_frame_spectral_l1(estimate, reference, n_fft=1024, hop=256):
    """[B, samples] x2 -> [B, T] mean absolute log-magnitude error."""
    window = torch.hann_window(n_fft, device=estimate.device)
    kw = dict(n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    est_mag = torch.log1p(torch.stft(estimate, **kw).abs())
    ref_mag = torch.log1p(torch.stft(reference, **kw).abs())
    return (est_mag - ref_mag).abs().mean(dim=1)       # average over freq


# ----------------------------------------------------------------------
# THE COMBINED LOSS - this is the contribution
# ----------------------------------------------------------------------
def impulse_weighted_loss(estimate, reference, noisy,
                          impulse_weight=3.0, spectral_alpha=1.0,
                          n_fft=1024, hop=256, threshold_db=8.0,
                          return_parts=False):
    """SI-SNR for overall quality + spectral error with impulse frames
    weighted `impulse_weight` times harder.

    Set impulse_weight=1.0 to get the plain baseline loss - that is
    your ablation study. Train both, compare, and the difference IS
    your result.
    """
    base = si_snr_loss(estimate, reference)

    frame_err = per_frame_spectral_l1(estimate, reference, n_fft, hop)
    impulses = detect_impulse_frames(noisy, n_fft, hop, threshold_db)

    # weight = 1 for normal frames, impulse_weight for impulse frames
    weights = 1.0 + (impulse_weight - 1.0) * impulses
    # normalise by total weight so the loss scale does not change with
    # how many impulses happen to be in the batch
    spectral = (frame_err * weights).sum() / (weights.sum() + 1e-8)

    total = base + spectral_alpha * spectral
    if return_parts:
        return total, {"si_snr_loss": base.detach(),
                       "spectral": spectral.detach(),
                       "impulse_frac": impulses.mean().detach()}
    return total


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 2, 48000
    ref = torch.randn(B, N) * 0.1
    est = ref + torch.randn(B, N) * 0.01               # slightly wrong
    noisy = ref.clone()
    noisy[:, 20000:20400] += 3.0                       # fake gunshot

    print("si_snr per item:", si_snr(est, ref).tolist())
    imp = detect_impulse_frames(noisy)
    print("impulse frames flagged:", int(imp.sum().item()), "of", imp.numel())
    total, parts = impulse_weighted_loss(est, ref, noisy, return_parts=True)
    print("total loss:", float(total))
    print("parts:", {k: float(v) for k, v in parts.items()})
