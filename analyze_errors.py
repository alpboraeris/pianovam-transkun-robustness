#!/usr/bin/env python3
"""
Error categorisation (section 3): which notes does TransKun miss,
and which notes does it invent? Reads existing predictions only, no TransKun runs.

Matching: pitch + onset within 50 ms (mir_eval, offsets ignored), the same rule as
the onset F1. Every reference note is then "found" or "missed", every predicted
note "correct" or "false positive".

Categories of reference notes (for recall)
  octave      A0-B1, C2-B2, ..., C7-C8
  register    low (A0-B2), mid (C3-B4), high (C5-C8)
  key colour  black / white, also within each register (registers differ in
              their share of black keys, so the pooled comparison alone can mislead)
  velocity    <40, 40-59, 60-79, >=80 (MIDI velocity of the reference note)
  polyphony   keys held down at the note's onset, including itself: 1, 2-3, 4-5, 6+
Same octave/register/key-colour categories for predicted notes (for precision).

Why a note was missed (checked in this order)
  timing      a prediction of the same pitch starts 50-200 ms away
  octave      a prediction 12 or 24 semitones away starts within 50 ms
  neighbour   a prediction 1-2 semitones away starts within 50 ms
  undetected  nothing of the above
Why a prediction is a false positive: same rules, looking for a reference note,
with "harmonic" (+-12, +-19, +-24 semitones) instead of "octave", and
"spurious" when nothing is near.

Groups
  clean          clean audio, all recordings of the split
  clean_dev      clean audio, only the recordings that also have noise results
  <type>@<snr>   noisy audio, e.g. music@-10 (pooled over recordings and excerpts)

Usage
  python analyze_errors.py --split valid
Outputs in results/analysis/errors/<split>/
  errors_by_category.csv, error_reasons.csv, notes_clean.csv,
  clean_error_categories.png, noise_recall_by_register.png, error_reasons.png
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from mir_eval import transcription as tr
from mir_eval.util import midi_to_hz

from evaluate_one import load_reference, load_estimate, intervals

BLACK = {1, 3, 6, 8, 10}
OCT_EDGES = [21, 36, 48, 60, 72, 84, 96, 109]
OCT_LABELS = ["A0-B1", "C2-B2", "C3-B3", "C4-B4", "C5-B5", "C6-B6", "C7-C8"]
VEL_EDGES, VEL_LABELS = [0, 40, 60, 80, 128], ["<40", "40-59", "60-79", ">=80"]
POLY_EDGES, POLY_LABELS = [1, 2, 4, 6, 1000], ["1", "2-3", "4-5", "6+"]
TYPE_OF = {"speech": "speech", "env": "environment", "music": "music", "white": "white"}
TYPES = ["speech", "environment", "music", "white"]

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REG_COLOR = {"low": "#86b6ef", "mid": "#2a78d6", "high": "#104281"}  # ordinal blue ramp
REG_MARK = {"low": "v", "mid": "o", "high": "^"}
REASON_COLOR = {"timing": "#2a78d6", "octave": "#eb6834", "harmonic": "#eb6834",
                "neighbour": "#1baf7a", "undetected": "#9a9890", "spurious": "#9a9890"}


def register(p):
    return np.where(p < 48, "low", np.where(p < 72, "mid", "high"))


def near(on_a, pitch_a, on_b_sorted, pitch_b_sorted_by_on, t0, t1, dpitches):
    """For each note in A: is there a note in B with onset distance in [t0, t1]
    and pitch difference in dpitches?"""
    out = np.zeros(len(on_a), dtype=bool)
    lo = np.searchsorted(on_b_sorted, on_a - t1, side="left")
    hi = np.searchsorted(on_b_sorted, on_a + t1, side="right")
    for i in range(len(on_a)):
        if hi[i] <= lo[i]:
            continue
        d_on = np.abs(on_b_sorted[lo[i]:hi[i]] - on_a[i])
        d_p = pitch_b_sorted_by_on[lo[i]:hi[i]] - pitch_a[i]
        ok = (d_on >= t0) & (d_on <= t1) & np.isin(d_p, dpitches)
        out[i] = ok.any()
    return out


def reasons(on_a, pitch_a, on_b, pitch_b, octave_set, octave_name, none_name):
    order = np.argsort(on_b)
    ob, pb = on_b[order], pitch_b[order]
    r = np.full(len(on_a), none_name, dtype=object)
    if len(on_a) == 0 or len(on_b) == 0:
        return r
    timing = near(on_a, pitch_a, ob, pb, 0.05, 0.2, [0])
    octv = near(on_a, pitch_a, ob, pb, 0.0, 0.05, octave_set)
    neigh = near(on_a, pitch_a, ob, pb, 0.0, 0.05, [-2, -1, 1, 2])
    r[neigh] = "neighbour"
    r[octv] = octave_name
    r[timing] = "timing"
    return r


def analyze(est_path, tsv_path):
    r_on, r_koff, _, r_pitch, r_vel = load_reference(tsv_path)
    e_on, e_off, e_pitch, _, _ = load_estimate(est_path)
    match = tr.match_notes(intervals(r_on, r_koff), midi_to_hz(r_pitch), intervals(e_on, e_off),
                           midi_to_hz(e_pitch), onset_tolerance=0.05, offset_ratio=None)
    r_hit = np.zeros(len(r_on), bool)
    e_hit = np.zeros(len(e_on), bool)
    for i, j in match:
        r_hit[i], e_hit[j] = True, True

    # polyphony: keys held (onset <= t < key release) at each reference onset
    starts, ends = np.sort(r_on), np.sort(r_koff)
    poly = np.searchsorted(starts, r_on, side="right") - np.searchsorted(ends, r_on, side="right")

    ref = pd.DataFrame({"pitch": r_pitch, "velocity": r_vel, "polyphony": poly, "found": r_hit})
    ref["reason"] = ""
    miss = ~r_hit
    ref.loc[miss, "reason"] = reasons(r_on[miss], r_pitch[miss], e_on, e_pitch, [-24, -12, 12, 24],
                                      "octave", "undetected")
    est = pd.DataFrame({"pitch": e_pitch, "correct": e_hit})
    est["reason"] = ""
    fp = ~e_hit
    est.loc[fp, "reason"] = reasons(e_on[fp], e_pitch[fp], r_on, r_pitch, [-24, -19, -12, 12, 19, 24],
                                    "harmonic", "spurious")
    for df in (ref, est):
        df["black"] = np.isin(df["pitch"] % 12, list(BLACK))
        df["key"] = np.where(df["black"], "black", "white")
        df["register"] = register(df["pitch"].values)
        df["octave"] = pd.cut(df["pitch"], OCT_EDGES, right=False, labels=OCT_LABELS).astype(str)
    ref["velocity_bin"] = pd.cut(ref["velocity"], VEL_EDGES, right=False, labels=VEL_LABELS).astype(str)
    ref["polyphony_bin"] = pd.cut(ref["polyphony"].clip(lower=1), POLY_EDGES, right=False,
                                  labels=POLY_LABELS).astype(str)
    return ref, est


def collect(split, data):
    """Yield (group, recording, noise_id, snr, est_path, tsv_path)."""
    for p in sorted(glob.glob(os.path.join("predictions", "clean", split, "*.mid"))):
        rid = os.path.splitext(os.path.basename(p))[0]
        yield "clean", rid, "clean", None, p, os.path.join(data, "TSV", rid + ".tsv")
    for p in sorted(glob.glob(os.path.join("predictions", "noise", split, "*", "*.mid"))):
        rid = os.path.basename(os.path.dirname(p))
        m = re.match(r"(.+)_snr([+-]\d+)\.mid$", os.path.basename(p))
        if not m:
            continue
        nid, snr = m.group(1), int(m.group(2))
        ntype = TYPE_OF.get(nid.split("_")[0], nid)
        yield f"{ntype}@{snr}", rid, nid, snr, p, os.path.join(data, "TSV", rid + ".tsv")


def summarize(ref, est, group):
    rows = []
    kinds = [("octave", "octave"), ("register", "register"), ("key", "key"),
             ("velocity", "velocity_bin"), ("polyphony", "polyphony_bin")]
    for kind, col in kinds:
        for cat, g in ref.groupby(col):
            e = est[est[col] == cat] if col in est.columns else None
            rows.append({"group": group, "kind": kind, "category": cat, "n_ref": len(g),
                         "recall": round(g["found"].mean(), 4),
                         "n_est": len(e) if e is not None else None,
                         "precision": round(e["correct"].mean(), 4) if e is not None and len(e) else None})
    for (reg, key), g in ref.groupby(["register", "key"]):
        e = est[(est["register"] == reg) & (est["key"] == key)]
        rows.append({"group": group, "kind": "register x key", "category": f"{reg} {key}", "n_ref": len(g),
                     "recall": round(g["found"].mean(), 4), "n_est": len(e),
                     "precision": round(e["correct"].mean(), 4) if len(e) else None})
    rows.append({"group": group, "kind": "all", "category": "all", "n_ref": len(ref),
                 "recall": round(ref["found"].mean(), 4), "n_est": len(est),
                 "precision": round(est["correct"].mean(), 4) if len(est) else None})
    return rows


def reason_rows(ref, est, group):
    rows = []
    n_ref = len(ref)
    for side, df, flag in (("missed", ref, "found"), ("false_positive", est, "correct")):
        bad = df[~df[flag]]
        for reason, g in bad.groupby("reason"):
            rows.append({"group": group, "side": side, "reason": reason, "count": len(g),
                         "per_1000_ref_notes": round(1000 * len(g) / n_ref, 2),
                         "share_of_side": round(len(g) / max(len(bad), 1), 4)})
    return rows


def style(ax, ylim=(0, 1.02)):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--data", default="data/PianoVAM_v1.1")
    a = ap.parse_args()
    out = os.path.join("results", "analysis", "errors", a.split)
    os.makedirs(out, exist_ok=True)

    refs, ests = {}, {}
    noisy_recs = set()
    for group, rid, nid, snr, p, tsv in collect(a.split, a.data):
        ref, est = analyze(p, tsv)
        for df in (ref, est):
            df["recording"], df["noise_id"], df["snr"] = rid, nid, snr
        refs.setdefault(group, []).append(ref)
        ests.setdefault(group, []).append(est)
        if group != "clean":
            noisy_recs.add(rid)
    if "clean" not in refs:
        raise SystemExit("No clean predictions found in predictions/clean/<split>/ (run run_clean.py first).")
    refs = {g: pd.concat(v, ignore_index=True) for g, v in refs.items()}
    ests = {g: pd.concat(v, ignore_index=True) for g, v in ests.items()}
    if noisy_recs:  # clean restricted to the recordings that have noise results
        refs["clean_dev"] = refs["clean"][refs["clean"]["recording"].isin(noisy_recs)]
        ests["clean_dev"] = ests["clean"][ests["clean"]["recording"].isin(noisy_recs)]

    def gkey(g):
        if g.startswith("clean"):
            return (0, g, 0)
        t, s = g.split("@")
        return (1 + TYPES.index(t) if t in TYPES else 9, t, -int(s))
    groups = sorted(refs, key=gkey)

    cat_rows, rea_rows = [], []
    for g in groups:
        cat_rows += summarize(refs[g], ests[g], g)
        rea_rows += reason_rows(refs[g], ests[g], g)
    cats, reas = pd.DataFrame(cat_rows), pd.DataFrame(rea_rows)
    cats.to_csv(os.path.join(out, "errors_by_category.csv"), index=False, encoding="utf-8-sig")
    reas.to_csv(os.path.join(out, "error_reasons.csv"), index=False, encoding="utf-8-sig")
    refs["clean"].to_csv(os.path.join(out, "notes_clean.csv"), index=False, encoding="utf-8-sig")

    # ---------------- console ----------------
    c = cats[cats.group == "clean"].set_index(["kind", "category"])
    n_rec = refs["clean"]["recording"].nunique()
    print(f"CLEAN audio, {n_rec} recordings, {len(refs['clean'])} reference notes, "
          f"recall {c.loc[('all', 'all'), 'recall']:.3f}, precision {c.loc[('all', 'all'), 'precision']:.3f}\n")
    for kind, labels in (("octave", OCT_LABELS), ("key", ["white", "black"]), ("velocity", VEL_LABELS),
                         ("polyphony", POLY_LABELS)):
        print(f"  recall by {kind}:")
        for lab in labels:
            if (kind, lab) in c.index:
                r = c.loc[(kind, lab)]
                print(f"    {lab:8s} {r['recall']:.3f}   (n={int(r['n_ref'])})")
    print("  recall by register x key colour:")
    for reg in ("low", "mid", "high"):
        parts = []
        for key in ("white", "black"):
            k = ("register x key", f"{reg} {key}")
            if k in c.index:
                parts.append(f"{key} {c.loc[k, 'recall']:.3f} (n={int(c.loc[k, 'n_ref'])})")
        print(f"    {reg:5s} " + "   ".join(parts))
    print("\n  why notes are missed / invented (clean):")
    for _, r in reas[reas.group == "clean"].iterrows():
        print(f"    {r['side']:15s} {r['reason']:11s} {int(r['count']):6d}  ({r['share_of_side']:.0%} of {r['side']})")

    if noisy_recs:
        print(f"\nNOISY audio ({len(noisy_recs)} recordings). Recall by register, compared with clean_dev:")
        cd = cats[cats.group == "clean_dev"].set_index(["kind", "category"])
        print(f"    {'group':16s} {'low':>6s} {'mid':>6s} {'high':>6s} {'all':>6s} {'prec':>6s}")
        for g in groups:
            if g == "clean":
                continue
            cg = cats[cats.group == g].set_index(["kind", "category"])
            vals = [cg.loc[("register", r), "recall"] if ("register", r) in cg.index else np.nan
                    for r in ("low", "mid", "high")]
            print(f"    {g:16s} " + " ".join(f"{v:6.3f}" for v in vals) +
                  f" {cg.loc[('all', 'all'), 'recall']:6.3f} {cg.loc[('all', 'all'), 'precision']:6.3f}")

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: clean error categories
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw={"width_ratios": [7, 3.2, 4, 4]})
    panels = [("octave", OCT_LABELS, "Recall by octave"), ("register x key", None, "Recall: white vs black keys"),
              ("velocity", VEL_LABELS, "Recall by velocity"), ("polyphony", POLY_LABELS, "Recall by polyphony")]
    lo = min(c["recall"].min() - 0.05, 0.85)
    for ax, (kind, labels, title) in zip(axes, panels):
        if kind == "register x key":
            for i, reg in enumerate(("low", "mid", "high")):
                for j, key in enumerate(("white", "black")):
                    k = ("register x key", f"{reg} {key}")
                    if k not in c.index:
                        continue
                    v, n = c.loc[k, "recall"], int(c.loc[k, "n_ref"])
                    x = i + (j - 0.5) * 0.38
                    ax.bar(x, v, 0.36, color="#d8d7d2" if key == "white" else "#3b3a37",
                           edgecolor=INK2, linewidth=0.6, label=key if i == 0 else None)
                    ax.text(x, v + 0.003, f"{n}", ha="center", va="bottom", fontsize=6, color=INK2, rotation=90)
            ax.set_xticks(range(3), ["low", "mid", "high"])
            ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        else:
            labs = [l for l in labels if (kind, l) in c.index]
            vals = [c.loc[(kind, l), "recall"] for l in labs]
            ns = [int(c.loc[(kind, l), "n_ref"]) for l in labs]
            ax.bar(range(len(labs)), vals, 0.7, color="#2a78d6", edgecolor=SURFACE, linewidth=1)
            for x, v, n in zip(range(len(labs)), vals, ns):
                ax.text(x, v + 0.003, f"n={n}", ha="center", va="bottom", fontsize=6.5, color=INK2)
            ax.set_xticks(range(len(labs)), labs, fontsize=7)
        style(ax, (lo, 1.01))
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    axes[0].set_ylabel("Recall (share of notes found)", fontsize=8, color=INK2)
    fig.suptitle(f"Which notes does TransKun miss? Clean audio, {a.split} split, {n_rec} recordings "
                 f"(y-axis starts at {lo:.2f})", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "clean_error_categories.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    if noisy_recs:
        # Fig 2: recall by register vs SNR, one panel per noise type
        snrs = sorted({int(g.split("@")[1]) for g in groups if "@" in g}, reverse=True)
        types = [t for t in TYPES if any(g.startswith(t + "@") for g in groups)]
        fig, axes = plt.subplots(1, len(types), figsize=(3.6 * len(types), 3.4), sharey=True, squeeze=False)
        xl = ["clean"] + ["0" if s == 0 else f"{s:+d}" for s in snrs]
        cd = cats[cats.group == "clean_dev"].set_index(["kind", "category"])
        for ax, t in zip(axes[0], types):
            for reg in ("low", "mid", "high"):
                ys = [cd.loc[("register", reg), "recall"]]
                for s in snrs:
                    cg = cats[cats.group == f"{t}@{s}"].set_index(["kind", "category"])
                    ys.append(cg.loc[("register", reg), "recall"] if ("register", reg) in cg.index else np.nan)
                ax.plot(range(len(xl)), ys, color=REG_COLOR[reg], marker=REG_MARK[reg], linewidth=2,
                        markersize=5, markeredgecolor=SURFACE, label=f"{reg} register")
            ax.set_xticks(range(len(xl)), xl)
            ax.set_title(t.capitalize() if t != "white" else "White noise", fontsize=9.5, color=INK, loc="left")
            ax.set_xlabel("SNR (dB)", fontsize=8, color=INK2)
            style(ax)
        axes[0][0].set_ylabel("Recall", fontsize=8, color=INK2)
        axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
        fig.suptitle(f"Recall by register under noise ({len(noisy_recs)} recordings; low A0-B2, mid C3-B4, "
                     f"high C5-C8)", fontsize=10.5, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "noise_recall_by_register.png"), dpi=150, facecolor=SURFACE)
        plt.close(fig)

        # Fig 3: why notes are missed / invented, per 1000 reference notes
        show = ["clean_dev"] + [f"{t}@{s}" for t in types for s in (0, -10) if f"{t}@{s}" in refs]
        fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=False)
        for ax, side, rs, title in (
                (axes[0], "missed", ["undetected", "octave", "neighbour", "timing"], "Missed notes, by reason"),
                (axes[1], "false_positive", ["spurious", "harmonic", "neighbour", "timing"],
                 "Invented notes (false positives), by reason")):
            bottom = np.zeros(len(show))
            for r in rs:
                vals = []
                for g in show:
                    m = reas[(reas.group == g) & (reas.side == side) & (reas.reason == r)]
                    vals.append(float(m["per_1000_ref_notes"].iloc[0]) if len(m) else 0.0)
                ax.bar(range(len(show)), vals, 0.7, bottom=bottom, color=REASON_COLOR[r], label=r,
                       edgecolor=SURFACE, linewidth=1)
                bottom += np.array(vals)
            ax.set_xticks(range(len(show)), [g.replace("clean_dev", "clean").replace("environment", "env.").replace("@", "\n") + ("" if g.startswith("clean") else " dB")
                                             for g in show], fontsize=7)
            style(ax, None)
            ax.set_title(title, fontsize=9.5, color=INK, loc="left")
            h, l = ax.get_legend_handles_labels()
            ax.legend(h[::-1], l[::-1], fontsize=7, frameon=False)  # same top-to-bottom order as the stack
        axes[0].set_ylabel("per 1000 reference notes", fontsize=8, color=INK2)
        fig.suptitle(f"Error types, clean vs noise at 0 and -10 dB ({len(noisy_recs)} recordings)",
                     fontsize=10.5, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "error_reasons.png"), dpi=150, facecolor=SURFACE)
        plt.close(fig)

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
