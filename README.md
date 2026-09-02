# 🛡️ ShabdKavach 

**Real-time speech enhancement for defence communications.**
Smart India Hackathon 2026.

---

## The problem

A soldier speaks into a radio. Gunfire, artillery and rotor noise get
mixed into the signal, and commands get misheard exactly when they
matter most.

Classical filters — spectral subtraction, Wiener filtering, LMS — all
assume the noise is steady. A gunshot lasts about 20 milliseconds and is
over before they can adapt.

## What this does

A small neural network looks at the noisy audio and decides, for every
frequency at every moment, how much of it is speech and how much is
noise. It keeps the speech and turns down the rest.

It runs **live**, on a **Rs 8,000 Raspberry Pi**, with **16 ms of
delay** — fast enough that the person on the other end hears you
normally.

```
mic -> spectrogram -> causal GRU -> mask (0 to 1) -> multiply -> audio out
```

The mask can only ever turn things **down**, never up. So the model can
degrade speech but it can never invent a word that was not spoken — an
important property for a system carrying orders.

---

## Results

Measured on held-out files the model never saw during training.

### It removes noise

| Noise type | Improvement (SI-SNR) |
|---|---|
| Steady — air conditioner | **+16.7 dB** |
| Impulsive — dog bark | **+12.7 dB** |
| Non-stationary — siren | **+6.7 dB** |
| Gunshot | **+5.4 dB** |
| Babble — crowd | +4.0 dB |

### It meets the problem statement targets

| Target | Achieved |
|---|---|
| SNR > 15 dB | **20.68 dB** |
| STOI > 0.85 | **0.943** |
| PESQ > 2.5 | **2.53** |

*(at 15 dB input SNR)*

### It beats a production model on signal fidelity

Against **DeepFilterNet3**, a published state-of-the-art system, on
identical files:

| Input SNR | DeepFilterNet3 | **ShabdKavach** |
|---|---|---|
| 10 dB | 16.60 dB | **17.34 dB** |
| 15 dB | 19.03 dB | **20.68 dB** |

We do this with **half the parameters** (1.1 M vs 2.1 M) and **60 minutes
of training audio** against their hundreds of hours.

### It runs in real time

| | |
|---|---|
| Model size | 4.5 MB |
| Latency | 16 ms |
| Real-time factor | 0.182 |
| Reconstruction error | 0.000000 |

---

## Try it

```bash
pip install torch torchaudio librosa soundfile pystoi sounddevice

# check the audio pipeline is exact - no model needed
python src/chain.py --check

# live microphone to headphone
python src/chain.py --live --ckpt checkpoints/best37.pt
```

To reproduce a result — mix two files and measure the improvement:

```bash
python src/spectrograms.py \
  --clean data/speech_test/<file>.wav \
  --noise data/noise_test/<file>.wav \
  --snr 0 --ckpt checkpoints/best37.pt --skip-dfn --save-audio \
  --out results/demo.png
```

---

## Honest limits

- Evaluation is held out at **file level**; speakers and noise source
  recordings do overlap, so the figures are optimistic.
- **ARM deployment is not yet measured** — timings are x86 CPU.
- Noise is **public proxy data**, not real defence recordings.
- The impulse-weighting ablation was **inconclusive** at clip level.

We state these because a number you cannot qualify is not a number.

---

## Built with

Python · PyTorch · torchaudio · librosa
LibriSpeech · UrbanSound8K · MUSAN

**Key references:** Bengio et al. 1994 (why gates are needed) ·
Chung et al. 2014 (GRU vs LSTM) · Valin 2018 (RNNoise) ·
Schroter et al. (DeepFilterNet)
