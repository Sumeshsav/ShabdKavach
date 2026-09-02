"""
plot_results.py - turn comparison.csv into figures for the deck.

Reads the CSV written by compare.py and produces:
  fig_metrics.png    STOI / SI-SNR / PESQ vs input SNR, one line per system
  fig_category.png   STOI by noise category at the hardest SNR

WHY THIS IS THE RIGHT PLOT
Speech enhancement results are conventionally shown as OUTPUT metric on
the y-axis against INPUT SNR on the x-axis. That shows how each system
behaves across conditions instead of one number at one operating point.

Two reference lines make the plot argue for you:
  - the "unprocessed" line is the floor. Any system below it is doing
    net harm at that input level.
  - on the SI-SNR panel, the dashed y = x diagonal marks "no change".
    A curve dipping below it means the system is DEGRADING the signal -
    this is exactly the crossover we measured on impulsive noise.

USAGE
  python plot_results.py --csv ../results/comparison.csv --out ../results
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Projector-legible defaults. Small fonts are the most common reason a
# good figure is useless on a slide.
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Distinct MARKERS as well as colours - never rely on colour alone,
# projectors and printed handouts wash colours out.
STYLES = [
    {"marker": "o", "linestyle": "--", "color": "#888888"},   # unprocessed
    {"marker": "s", "linestyle": "-", "color": "#1f77b4"},
    {"marker": "^", "linestyle": "-", "color": "#d62728"},
    {"marker": "D", "linestyle": "-", "color": "#2ca02c"},
]

TARGETS = {"stoi": (0.85, "PS target 0.85"),
           "pesq": (2.5, "PS target 2.5")}


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["snr_db"] = float(r["snr_db"])
            for k in ("stoi", "si_snr", "pesq", "rtf"):
                r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
            rows.append(r)
    return rows


def mean_of(rows, system, snr, key, cat=None):
    vals = [r[key] for r in rows
            if r["system"] == system and r["snr_db"] == snr
            and r[key] is not None and (cat is None or r["category"] == cat)]
    return float(np.mean(vals)) if vals else None


def order_systems(rows):
    """Unprocessed first so it reads as the baseline floor."""
    names = list(dict.fromkeys(r["system"] for r in rows))
    names.sort(key=lambda n: (0 if "nprocessed" in n else 1, n))
    return names


def fig_metrics(rows, outdir):
    systems = order_systems(rows)
    snrs = sorted({r["snr_db"] for r in rows})

    # only plot panels that actually have data
    panels = [("stoi", "STOI (0-1)", "Intelligibility"),
              ("si_snr", "output SI-SNR (dB)", "Signal fidelity")]
    if any(r["pesq"] is not None for r in rows):
        panels.append(("pesq", "PESQ (wideband)", "Perceptual quality"))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (key, ylabel, title) in zip(axes, panels):
        for i, sysname in enumerate(systems):
            ys = [mean_of(rows, sysname, s, key) for s in snrs]
            if all(y is None for y in ys):
                continue
            st = STYLES[i % len(STYLES)]
            ax.plot(snrs, ys, label=sysname, markersize=8,
                    linewidth=2, **st)

        # y = x reference on the SI-SNR panel: below this line the
        # system is making the signal WORSE than it received.
        if key == "si_snr":
            lo, hi = min(snrs), max(snrs)
            ax.plot([lo, hi], [lo, hi], ":", color="#c1654f", linewidth=1.8,
                    label="no change (output = input)")

        if key in TARGETS:
            val, lbl = TARGETS[key]
            ax.axhline(val, linestyle=":", color="#c1654f", linewidth=1.8)
            ax.text(snrs[0], val, " " + lbl, va="bottom", ha="left",
                    fontsize=10, color="#c1654f")

        ax.set_xlabel("input SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(snrs)
        ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(outdir, "fig_metrics.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_category(rows, outdir):
    cats = sorted({r["category"] for r in rows})
    if len(cats) < 2:
        print("only one noise category - skipping category figure")
        return
    systems = order_systems(rows)
    snr = min(r["snr_db"] for r in rows)          # hardest condition

    fig, ax = plt.subplots(figsize=(1.9 * len(cats) * len(systems) + 3, 5))
    width = 0.8 / len(systems)
    x = np.arange(len(cats))

    for i, sysname in enumerate(systems):
        ys = [mean_of(rows, sysname, snr, "stoi", c) or 0 for c in cats]
        ax.bar(x + i * width - 0.4 + width / 2, ys, width,
               label=sysname, color=STYLES[i % len(STYLES)]["color"],
               edgecolor="white")
        for xi, y in zip(x + i * width - 0.4 + width / 2, ys):
            ax.text(xi, y + 0.008, f"{y:.3f}", ha="center", fontsize=9)

    ax.axhline(0.85, linestyle=":", color="#c1654f", linewidth=1.8)
    ax.text(-0.45, 0.85, " PS target 0.85", va="bottom", fontsize=10,
            color="#c1654f")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_xlabel("noise category")
    ax.set_ylabel("STOI (0-1)")
    ax.set_title(f"Intelligibility by noise type at {snr:.0f} dB input SNR")
    ax.legend(loc="best")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_category.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="../results/comparison.csv")
    p.add_argument("--out", default="../results")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = load(args.csv)
    print(f"loaded {len(rows)} rows, "
          f"{len(set(r['system'] for r in rows))} systems")
    fig_metrics(rows, args.out)
    fig_category(rows, args.out)
    print("\nBoth figures are sized and styled for a projector.")
    print("Drop fig_metrics.png on slide 4 and fig_category.png on slide 2.")
