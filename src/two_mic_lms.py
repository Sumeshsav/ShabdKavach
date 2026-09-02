"""
two_mic_lms.py - does LMS actually work when the reference is clean?

WHY THIS EXISTS
The single-microphone version failed because the "reference" was made
from the model's own leftover, which still contained speech. The filter
subtracted the speech along with the noise.

This script removes that doubt by SIMULATING the two-microphone setup
the problem statement assumes:

    PRIMARY   mic at the mouth      = speech + noise
    REFERENCE mic facing outward    = noise only, no speech

We already have speech and noise as separate files, so we can build both
channels exactly. The reference passes through a different, unknown path
(delay + attenuation + a little filtering) so the filter has something
real to learn - just as two microphones in different positions would.

If LMS works here, the algorithm is fine and the only thing missing is a
second microphone. If it fails here too, the approach is wrong.

USAGE
  python two_mic_lms.py --clean ../data/speech_test/<a.wav> \\
                        --noise ../data/noise_test/<gunshot.wav> --snr 0
"""

import argparse
import numpy as np
import librosa
import soundfile as sf

SR = 48000


# ----------------------------------------------------------------------
def si_snr(ref, est, eps=1e-8):
    ref = ref - ref.mean()
    est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    tgt = a * ref
    return 10 * np.log10((np.sum(tgt ** 2) + eps) /
                         (np.sum((est - tgt) ** 2) + eps))


def stoi_of(ref, est):
    try:
        from pystoi import stoi
        n = min(len(ref), len(est))
        return stoi(ref[:n], est[:n], SR, extended=False)
    except Exception:
        return None


# ----------------------------------------------------------------------
def make_two_channels(speech, noise, snr_db, delay=37, atten=0.62):
    """Build what two microphones would actually capture.

    Physics that matters: the REFERENCE mic sits closer to the noise
    source, so the noise reaches it FIRST. It arrives at the primary a
    little later, quieter, and slightly coloured by the path between.

    That ordering is essential. A causal filter can only look at PAST
    reference samples, so the reference must LEAD the primary. Build it
    the other way round and the filter would need to see the future,
    and it fails - it adds noise instead of removing it.

        REFERENCE = noise, as it arrives at the outward-facing mic
        PRIMARY   = speech + that same noise, delayed and filtered
    """
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    noise = noise[:len(speech)]

    ps, pn = np.mean(speech ** 2), np.mean(noise ** 2) + 1e-12
    alpha = np.sqrt(ps / (pn * (10 ** (snr_db / 10))))
    noise = alpha * noise

    reference = noise.copy()                    # noise-only mic

    # the same noise, later and quieter, as heard at the mouth mic
    at_primary = np.roll(noise, delay) * atten
    at_primary[:delay] = 0.0
    at_primary = np.convolve(at_primary, np.array([0.6, 0.3, 0.1]),
                             mode="same")

    primary = speech + at_primary

    return (primary.astype(np.float32), reference.astype(np.float32),
            at_primary.astype(np.float32))


def nlms(primary, reference, taps=128, mu=0.25, eps=1e-6):
    """Textbook normalised LMS, time domain.

        y[n] = w . x[n]            filter output, an estimate of the noise
        e[n] = d[n] - y[n]         what is left after subtracting it
        w   += mu * e[n] * x[n] / (||x[n]||^2 + eps)

    d is the primary mic, x is the reference. The error signal e IS the
    cleaned output: whatever the filter could not explain from the
    reference is, by definition, the speech.
    """
    n = len(primary)
    w = np.zeros(taps, dtype=np.float64)
    x = np.zeros(taps, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        x[1:] = x[:-1]
        x[0] = reference[i]
        y = float(np.dot(w, x))
        e = primary[i] - y
        out[i] = e
        p = float(np.dot(x, x)) + eps
        w += (mu * e / p) * x
    return out.astype(np.float32), w


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stereo", help="real 2-channel recording; skips the "
                                    "simulation and uses it directly")
    p.add_argument("--primary-ch", type=int, default=0)
    p.add_argument("--ref-ch", type=int, default=1)
    p.add_argument("--clean")
    p.add_argument("--noise")
    p.add_argument("--snr", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--taps", type=int, default=128)
    p.add_argument("--mu", type=float, default=0.005,
                   help="step size. Smaller converges slower "
                        "but settles closer to the true path")
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    # ---- real two-channel recording -------------------------------
    if args.stereo:
        x, sr = sf.read(args.stereo, always_2d=True)
        if sr != SR:
            x = np.stack([librosa.resample(x[:, c], orig_sr=sr,
                                           target_sr=SR)
                          for c in range(x.shape[1])], axis=1)
        primary = x[:, args.primary_ch].astype(np.float32)
        reference = x[:, args.ref_ch].astype(np.float32)
        n = min(len(primary), len(reference))
        primary, reference = primary[:n], reference[:n]

        corr = abs(np.corrcoef(primary.astype(np.float64),
                               reference.astype(np.float64))[0, 1])
        print(f"\nreal recording: {args.stereo}")
        print(f"primary = ch{args.primary_ch}, reference = ch{args.ref_ch}")
        print(f"inter-channel correlation : {corr:.4f}")
        if corr > 0.98:
            print("  channels are near-identical - nothing for the filter "
                  "to exploit")

        out, w = nlms(primary, reference, args.taps, args.mu)
        sf.write("twomic_real_out.wav", out, SR)
        print(f"\nin  rms {np.sqrt(np.mean(primary**2)):.5f}")
        print(f"out rms {np.sqrt(np.mean(out**2)):.5f}")
        print(f"||w|| = {np.linalg.norm(w):.3f}")
        print("wrote twomic_real_out.wav - LISTEN to it.")
        print("\nNo clean reference exists for a real recording, so there")
        print("is no SI-SNR or STOI to report. Your ears are the measure:")
        print("did the noise drop while your speech survived?")
        return

    if not args.clean or not args.noise:
        print("Need --clean and --noise, or --stereo")
        return

    seg = int(args.seconds * SR)
    speech, _ = librosa.load(args.clean, sr=SR, mono=True)
    speech = np.pad(speech, (0, max(0, seg - len(speech))))[:seg]
    noise, _ = librosa.load(args.noise, sr=SR, mono=True)

    primary, reference, at_primary = make_two_channels(speech, noise,
                                                       args.snr)

    print(f"\nspeech : {args.clean.split('/')[-1]}")
    print(f"noise  : {args.noise.split('/')[-1]}")
    print(f"mixed at {args.snr:+.0f} dB SNR, {args.seconds:.0f} s\n")

    # sanity: the reference must contain NO speech
    corr = abs(np.corrcoef(reference, speech)[0, 1])
    print(f"reference-vs-speech correlation : {corr:.4f}"
          f"   {'(clean, as intended)' if corr < 0.1 else '(CONTAMINATED)'}")

    out, w = nlms(primary, reference, args.taps, args.mu)

    print(f"\n{'':22}{'SI-SNR':>10}{'STOI':>9}")
    print("-" * 41)
    for name, sig in [("primary (noisy)", primary), ("after two-mic LMS", out)]:
        s = si_snr(speech, sig)
        t = stoi_of(speech.astype(np.float64), sig.astype(np.float64))
        print(f"{name:22}{s:>10.2f}{(f'{t:.3f}' if t else 'n/a'):>9}")

    gain = si_snr(speech, out) - si_snr(speech, primary)
    print("-" * 41)
    print(f"{'improvement':22}{gain:>+10.2f} dB")
    print(f"\nfilter converged, ||w|| = {np.linalg.norm(w):.3f}, "
          f"{args.taps} taps")

    if args.save:
        sf.write("twomic_primary.wav", primary, SR)
        sf.write("twomic_reference.wav", reference, SR)
        sf.write("twomic_output.wav", out, SR)
        print("wrote twomic_primary.wav / _reference.wav / _output.wav")

    print("\nReading this: a large positive improvement means the LMS stage")
    print("works whenever the reference is genuinely speech-free. The only")
    print("thing missing in hardware is the second microphone.")


if __name__ == "__main__":
    main()
