#!/usr/bin/env python3
"""
Diagrams for the noise experiment (reads results/noise/<split>/noise_results.csv).

Outputs in results/noise/<split>/plots/
  f1_vs_snr_mean.png          F1 vs SNR per noise type, mean over recordings;
                              panels: onset, onset+offset (key release), frame (key release)
  f1_vs_snr_by_recording.png  same per recording (onset and onset+offset)
  f1_bars_onset.png           grouped bars: onset F1 per SNR level and noise type
  summary_by_type_snr.csv     the numbers behind the plots, incl. drop vs clean

Usage
  python plot_noise.py --split valid
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TYPES = ["speech", "environment", "music", "white"]
LABEL = {"speech": "Speech", "environment": "Environment", "music": "Music", "white": "White noise"}
COLOR = {"speech": "#2a78d6", "environment": "#eb6834", "music": "#1baf7a", "white": "#eda100"}  # validated
MARKER = {"speech": "o", "environment": "s", "music": "^", "white": "D"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
METRICS = [("onset", "Onset F1"), ("on+off_key", "Onset + offset F1 (key release)"),
           ("frame_key", "Frame F1 (key release)")]


def fmt(s):
    return "0" if s == 0 else f"{s:+d}"


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["snr_db"] = None if r["snr_db"] in ("", "None") else int(float(r["snr_db"]))
        for k, _ in METRICS:
            r[k] = float(r[k])
    return rows


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--data", default="data/PianoVAM_v1.1")
    a = ap.parse_args()

    base = os.path.join("results", "noise", a.split)
    out = os.path.join(base, "plots")
    os.makedirs(out, exist_ok=True)
    rows = load(os.path.join(base, "noise_results.csv"))

    names = {}
    try:
        with open(os.path.join(a.data, "manifest.json"), encoding="utf-8") as f:
            for r in json.load(f)["recordings"]:
                names[r["record_time"]] = f"{r.get('composer')}: {r.get('piece')}"
    except OSError:
        pass

    noisy = [r for r in rows if r["noise_type"] != "clean"]
    recs = sorted({r["record_time"] for r in noisy})             # only recordings with noise results
    clean = {r["record_time"]: r for r in rows if r["noise_type"] == "clean" and r["record_time"] in recs}
    snrs = sorted({r["snr_db"] for r in noisy}, reverse=True)    # +20 ... -20
    types = [t for t in TYPES if any(r["noise_type"] == t for r in noisy)]
    xs = list(range(len(snrs) + 1))                              # x=0 is clean
    xlabels = ["clean"] + [fmt(s) for s in snrs]

    # per (type, snr, recording): mean over excerpts of that type
    cell = defaultdict(list)
    for r in noisy:
        cell[(r["noise_type"], r["snr_db"], r["record_time"])].append(r)

    def value(t, snr, rec, k):
        v = cell.get((t, snr, rec))
        return np.mean([x[k] for x in v]) if v else np.nan

    def mean_over_recs(t, snr, k):
        vals = [value(t, snr, rec, k) for rec in recs]
        vals = [v for v in vals if not np.isnan(v)]
        return (np.mean(vals), len(vals)) if vals else (np.nan, 0)

    clean_mean = {k: np.mean([clean[r][k] for r in recs if r in clean]) for k, _ in METRICS}

    # ---- summary table ----
    table = []
    for t in types:
        for snr in snrs:
            row = {"noise_type": t, "snr_db": snr}
            for k, _ in METRICS:
                m, n = mean_over_recs(t, snr, k)
                row[f"{k}_mean"] = round(float(m), 4)
                row[f"{k}_drop_vs_clean"] = round(float(clean_mean[k] - m), 4)
                row["n_recordings"] = n
            table.append(row)
    with open(os.path.join(out, "summary_by_type_snr.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    # ---- figure 1: mean F1 vs SNR ----
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.6), sharey=True)
    for ax, (k, title) in zip(axes, METRICS):
        for t in types:
            ys = [clean_mean[k]] + [mean_over_recs(t, s, k)[0] for s in snrs]
            ax.plot(xs, ys, color=COLOR[t], linewidth=2, marker=MARKER[t], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABEL[t])
        ax.set_xticks(xs, xlabels)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("SNR (dB)   ← cleaner        noisier →", fontsize=8, color=INK2)
        style(ax)
    axes[0].set_ylabel("F1 (mean over recordings)", fontsize=8, color=INK2)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(f"TransKun under additive noise, {a.split} split, {len(recs)} recording(s)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "f1_vs_snr_mean.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- figure 2: per recording ----
    mets = METRICS[:2]
    fig, axes = plt.subplots(len(recs), len(mets), figsize=(4.6 * len(mets), 2.7 * len(recs)),
                             sharex=True, sharey=True, squeeze=False)
    for i, rec in enumerate(recs):
        for j, (k, title) in enumerate(mets):
            ax = axes[i][j]
            for t in types:
                ys = [clean[rec][k] if rec in clean else np.nan] + [value(t, s, rec, k) for s in snrs]
                ax.plot(xs, ys, color=COLOR[t], linewidth=2, marker=MARKER[t], markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=1, label=LABEL[t])
            ax.set_xticks(xs, xlabels)
            style(ax)
            ax.set_title(f"{names.get(rec, rec)}\n{rec} · {title}", fontsize=8, color=INK, loc="left")
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("SNR (dB)", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "f1_vs_snr_by_recording.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- figure 3: grouped bars, onset ----
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    width = 0.8 / len(types)
    for i, t in enumerate(types):
        ys = [mean_over_recs(t, s, "onset")[0] for s in snrs]
        pos = np.arange(len(snrs)) + (i - (len(types) - 1) / 2) * width
        ax.bar(pos, ys, width=width - 0.02, color=COLOR[t], label=LABEL[t], edgecolor=SURFACE, linewidth=1)
    ax.axhline(clean_mean["onset"], color=INK2, linewidth=1, linestyle="--")
    ax.text(len(snrs) - 0.45, clean_mean["onset"] + 0.01, f"clean {clean_mean['onset']:.3f}",
            fontsize=8, color=INK2, ha="right", va="bottom")
    ax.set_xticks(np.arange(len(snrs)), [f"{fmt(s)} dB" for s in snrs])
    style(ax)
    ax.set_ylabel("Onset F1 (mean over recordings)", fontsize=8, color=INK2)
    ax.legend(fontsize=8, frameon=False, ncol=len(types), loc="upper right", bbox_to_anchor=(1, 1.13))
    ax.set_title("Onset F1 by noise type and SNR", fontsize=10, color=INK, loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "f1_bars_onset.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- console ----
    print(f"Recordings: {', '.join(recs)}")
    print(f"\nOnset F1, mean over recordings (clean = {clean_mean['onset']:.3f})")
    print(f"{'noise':12s} " + " ".join(f"{s:>+7d}" for s in snrs))
    for t in types:
        print(f"{LABEL[t]:12s} " + " ".join(f"{mean_over_recs(t, s, 'onset')[0]:7.3f}" for s in snrs))
    print(f"\nOnset + offset F1 (key release), mean (clean = {clean_mean['on+off_key']:.3f})")
    for t in types:
        print(f"{LABEL[t]:12s} " + " ".join(f"{mean_over_recs(t, s, 'on+off_key')[0]:7.3f}" for s in snrs))
    print(f"\nSaved to {out}: f1_vs_snr_mean.png, f1_vs_snr_by_recording.png, f1_bars_onset.png, "
          f"summary_by_type_snr.csv")


if __name__ == "__main__":
    main()
