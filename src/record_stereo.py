"""
record_stereo.py - capture two microphone channels and find out whether
they are actually different.

Before any two-mic processing can work you need to know three things:

  1. Do the two channels carry DIFFERENT signals, or has Windows already
     beamformed them into one? If the inter-channel correlation is above
     about 0.98 they are effectively the same signal and there is
     nothing for an adaptive filter to work with.

  2. Does the reference channel contain your SPEECH? It must not. Two
     mics a few centimetres apart on a laptop both hear you clearly, so
     expect this to fail on built-in arrays.

  3. Which channel is which?

USAGE
  python record_stereo.py --list
  python record_stereo.py --device 15 --seconds 8 --out ../stereo.wav

  # then, while it records:
  #   first 4 s - speak, no noise
  #   last  4 s - stay silent, play the gunshot near ONE side
"""

import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 48000


def analyse(x, sr=SR, label=""):
    a, b = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    ra, rb = np.sqrt(np.mean(a ** 2)), np.sqrt(np.mean(b ** 2))
    corr = float(np.corrcoef(a, b)[0, 1]) if ra > 1e-9 and rb > 1e-9 else 0.0
    print(f"  {label:16} ch0 rms {ra:.5f}   ch1 rms {rb:.5f}   "
          f"corr {corr:+.4f}")
    return ra, rb, corr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--device", type=int)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--out", default="../stereo.wav")
    args = p.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    info = sd.query_devices(args.device)
    print(f"device {args.device}: {info['name']}")
    print(f"input channels available: {info['max_input_channels']}")
    if info["max_input_channels"] < 2:
        print("This device gives only one channel. Pick a 2-in device.")
        return

    print(f"\nRecording {args.seconds:.0f} s on 2 channels.")
    print("  first half : SPEAK, no noise")
    print("  second half: stay SILENT, play noise close to one side\n")
    x = sd.rec(int(args.seconds * SR), samplerate=SR, channels=2,
               dtype="float32", device=args.device)
    sd.wait()
    sf.write(args.out, x, SR)
    print(f"wrote {args.out}\n")

    half = len(x) // 2
    print("ANALYSIS")
    analyse(x, label="whole take")
    _, _, c_sp = analyse(x[:half], label="speech half")
    _, _, c_no = analyse(x[half:], label="noise half")

    print("\nWHAT THIS MEANS")
    whole_corr = float(np.corrcoef(x[:, 0].astype(np.float64),
                                   x[:, 1].astype(np.float64))[0, 1])
    if abs(whole_corr) > 0.98:
        print("  Channels are near-identical. Windows has almost certainly")
        print("  beamformed the array into one signal, or is duplicating a")
        print("  mono capture. There is no second observation to exploit -")
        print("  two-mic LMS cannot work on this device.")
    elif abs(c_sp) > 0.8:
        print("  Both channels hear your speech strongly (corr "
              f"{c_sp:+.2f} on the speech half).")
        print("  Neither channel is a speech-free reference, so LMS would")
        print("  cancel speech. Expected for mics centimetres apart.")
        print("  A usable reference must be POSITIONED to reject speech:")
        print("  facing away, on the shoulder or the far side of a helmet.")
    else:
        print("  The channels differ and the speech is not equally present")
        print("  in both. Worth trying:")
        print(f"    python two_mic_lms.py --stereo {args.out}")

    print("\n  Speech-half correlation is the number that matters. Close to")
    print("  zero means one channel is genuinely a noise reference.")


if __name__ == "__main__":
    main()
