"""
train.py - the training loop.

WHAT TRAINING ACTUALLY IS
  1. take a batch of (noisy, clean) pairs
  2. run the model on the noisy audio -> a guess
  3. score how wrong the guess was          (the loss)
  4. compute which direction every weight should move to make the loss
     smaller                                 (loss.backward())
  5. nudge the weights that way              (optimizer.step())
Repeat a few hundred thousand times.

RUN
  python src/train.py --speech data/speech --noise data/noise --epochs 30

  # ablation: same run with the impulse weighting switched off
  python src/train.py ... --impulse-weight 1.0 --out checkpoints/baseline
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import NoisySpeechDataset
from model import MaskNet
from losses import si_snr, impulse_weighted_loss


def run_epoch(model, loader, device, optimizer=None,
              impulse_weight=3.0, spectral_alpha=1.0):
    """One pass over the data. Pass optimizer=None for validation."""
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "si_snr": 0.0, "n": 0}

    # torch.enable_grad/no_grad: gradients are only needed for training.
    # Turning them off for validation saves memory and time.
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)

            estimate = model(noisy)
            loss = impulse_weighted_loss(
                estimate, clean, noisy,
                impulse_weight=impulse_weight,
                spectral_alpha=spectral_alpha,
            )

            if training:
                optimizer.zero_grad()        # clear last step's gradients
                loss.backward()              # compute new ones
                # Clip gradients: RNNs can produce huge gradients that
                # blow the weights up. Capping the norm keeps training
                # stable. Standard practice, not a hack.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()             # apply the nudge

            bs = noisy.shape[0]
            totals["loss"] += float(loss.detach()) * bs
            totals["si_snr"] += float(si_snr(estimate, clean).mean().detach()) * bs
            totals["n"] += bs

    n = max(totals["n"], 1)
    return totals["loss"] / n, totals["si_snr"] / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--speech", required=True)
    p.add_argument("--noise", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--segment", type=float, default=4.0, help="seconds")
    p.add_argument("--train-size", type=int, default=2000,
                   help="pairs per epoch (mixing is random, so this is free)")
    p.add_argument("--val-size", type=int, default=200)
    p.add_argument("--impulse-weight", type=float, default=3.0,
                   help="1.0 disables the weighting -> baseline ablation")
    p.add_argument("--spectral-alpha", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    print(f"device: {device}")

    # seed=0 for train, seed=1 for val so the two never draw the same
    # random crops. Not a perfect speaker split - for a proper paper you
    # would hold out whole SPEAKERS, not just different random crops.
    train_ds = NoisySpeechDataset(args.speech, args.noise,
                                  segment_seconds=args.segment,
                                  length=args.train_size, seed=0)
    val_ds = NoisySpeechDataset(args.speech, args.noise,
                                segment_seconds=args.segment,
                                length=args.val_size, seed=1)

    # num_workers=0 keeps it simple and avoids Windows multiprocessing
    # pain. Raise it on Colab/Linux if data loading is the bottleneck.
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    model = MaskNet(hidden=args.hidden, layers=args.layers).to(device)
    print(f"model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Drop the learning rate when validation stops improving - lets the
    # model settle into a good solution instead of bouncing around it.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    best = float("-inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_snr = run_epoch(model, train_dl, device, optimizer,
                                    args.impulse_weight, args.spectral_alpha)
        va_loss, va_snr = run_epoch(model, val_dl, device, None,
                                    args.impulse_weight, args.spectral_alpha)
        scheduler.step(va_snr)

        flag = ""
        if va_snr > best:
            best = va_snr
            torch.save({"model": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "val_si_snr": va_snr},
                       os.path.join(args.out, "best.pt"))
            flag = "  <- saved"

        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train loss {tr_loss:7.3f}  SI-SNR {tr_snr:6.2f} dB  |  "
              f"val loss {va_loss:7.3f}  SI-SNR {va_snr:6.2f} dB  "
              f"({time.time()-t0:.0f}s){flag}")

    print(f"\nbest validation SI-SNR: {best:.2f} dB")
    print(f"checkpoint: {os.path.join(args.out, 'best.pt')}")
    print("\nNext: run evaluate.py with this checkpoint and compare against")
    print("your DeepFilterNet baseline on the SAME test files.")


if __name__ == "__main__":
    main()
