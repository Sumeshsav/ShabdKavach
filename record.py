"""
Record a 48 kHz mono WAV - the exact format DeepFilterNet wants.

  pip install sounddevice

  python record.py clean.wav 10      # record 10 seconds
  python record.py noise.wav 5       # record 5 seconds of noise
"""

import sys
import sounddevice as sd
import soundfile as sf

SR = 48000

name = sys.argv[1] if len(sys.argv) > 1 else "clean.wav"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10

print(f"Recording {seconds:.0f}s to {name} ... speak now")
audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
sd.wait()
sf.write(name, audio, SR)
print(f"Saved {name}")
