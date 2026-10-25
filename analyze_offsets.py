#!/usr/bin/env python3
"""
Note-ending (offset) diagnostic for section 4.

Reads existing TransKun predictions; does not run TransKun.

For every recording of a split:
  1. Match predicted notes to reference notes on pitch + onset (50 ms), ignoring
     offsets, so every correctly found note gets an offset error.
  2. Signed offset error = predicted end - key release (TSV key_offset).
     Positive = TransKun ends the note too late, negative = too early.
  3. Pedal state at the moment of key release, read from the MIDI CC64 track
     (down = value >= 64). Notes released with the pedal up and with the pedal
     down are compared *within the same recording*, which holds piece,
     performer and recording session constant.
  4. Half-pedal check: share of key releases with 0 < CC64 < 64.
  5. TransKun's predicted pedal vs the real pedal, frame-wise at 60 Hz.

Usage
  python analyze_offsets.py --split valid

Outputs (results/analysis/<split>/)
  offsets_by_recording.csv   one row per recording
  offsets_notes.csv          one row per matched note (for histograms later)
  offset_error_hist.png      signed offset error, pedal up vs pedal down
"""

import argparse
import csv
import json
import os

import numpy as np
import pretty_midi
from mir_eval import transcription as tr
from mir_eval.util import midi_to_hz

from evaluate_one import load_reference, load_estimate, intervals

FRAME_RATE = 60
TOL_MIN, TOL_RATIO = 0.05, 0.2  # offset tolerance of the brief: max(50 ms, 20% of duration)
BLUE, ORANGE = "#2a78d6", "#eb6834"  # validated categorical pair (light surface)


def cc64_timeline(midi_path):
    """Return (times, values) of the real sustain pedal from the performance MIDI."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    ccs = sorted((c.time, c.value) for inst in pm.instruments
                 for c in inst.control_changes if c.number == 64)
    if not ccs:
        return np.array([0.0]), np.array([0])
    t, v = zip(*ccs)
    return np.array(t), np.array(v)


def cc_value_at(times, values, query):
    """CC64 value in effect at each query time (0 before the first event)."""
    idx = np.searchsorted(times, query, side="right") - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, None)], 0)
    return out


def pedal_mask(times, values, n_frames):
    """Boolean pedal-down mask on the 60 Hz grid from a CC64 timeline."""
    t = np.arange(n_frames) / FRAME_RATE
    return cc_value_at(times, values, t) >= 64


def predicted_pedal(est_midi, n_frames):
    pm = pretty_midi.PrettyMIDI(est_midi)
    ccs = sorted((c.time, c.value) for inst in pm.instruments
                 for c in inst.control_changes if c.number == 64)
    if not ccs:
        return np.zeros(n_frames, dtype=bool)
    t, v = zip(*ccs)
    return pedal_mask(np.array(t), np.array(v), n_frames)


def prf(ref, est):
    tp = np.logical_and(ref, est).sum()
    p = tp / est.sum() if est.sum() else float("nan")
    r = tp / ref.sum() if ref.sum() else float("nan")
    f = 2 * p * r / (p + r) if (p + r) and not np.isnan(p + r) else float("nan")
    return p, r, f


def summarize(err, ref_dur):
    """Stats for a group of notes; err in seconds."""
    if len(err) == 0:
        return {"n": 0}
    tol = np.maximum(TOL_RATIO * ref_dur, TOL_MIN)
    return {
        "n": int(len(err)),
        "median_ms": round(float(np.median(err)) * 1000, 1),
        "median_abs_ms": round(float(np.median(np.abs(err))) * 1000, 1),
        "share_late": round(float(np.mean(err > tol)), 3),
        "share_early": round(float(np.mean(err < -tol)), 3),
        "share_within_tol": round(float(np.mean(np.abs(err) <= tol)), 3),
    }


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/PianoVAM_v1.1")
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    a = ap.parse_args()

    with open(os.path.join(a.data, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    meta = {m["record_time"]: m for m in (meta.values() if isinstance(meta, dict) else meta)}
    with open(os.path.join(a.data, "manifest.json"), encoding="utf-8") as f:
        recs = [r for r in json.load(f)["recordings"] if r["split"] == a.split]

    pred_dir = os.path.join("predictions", "clean", a.split)
    out_dir = os.path.join("results", "analysis", a.split)
    os.makedirs(out_dir, exist_ok=True)

    rows, note_rows, plot_data = [], [], []
    for r in recs:
        rid = r["record_time"]
        est_path = os.path.join(pred_dir, rid + ".mid")
        if not os.path.exists(est_path):
            print(f"  missing prediction for {rid}, skipped (run run_clean.py first)")
            continue
        r_on, r_koff, r_foff, r_pitch, _ = load_reference(os.path.join(a.data, "TSV", rid + ".tsv"))
        e_on, e_off, e_pitch, _, _ = load_estimate(est_path)
        ct, cv = cc64_timeline(os.path.join(a.data, "MIDI", rid + ".mid"))

        # onset-only matching, so offsets can be studied for every found note
        match = tr.match_notes(intervals(r_on, r_koff), midi_to_hz(r_pitch),
                               intervals(e_on, e_off), midi_to_hz(e_pitch),
                               onset_tolerance=0.05, offset_ratio=None)
        ri = np.array([m[0] for m in match], dtype=int)
        ei = np.array([m[1] for m in match], dtype=int)

        err = e_off[ei] - r_koff[ri]                 # vs key release
        err_ped = e_off[ei] - r_foff[ri]             # vs pedal-extended ending
        dur = r_koff[ri] - r_on[ri]
        cc_at_release = cc_value_at(ct, cv, r_koff[ri])
        down = cc_at_release >= 64
        half = (cc_at_release > 0) & (cc_at_release < 64)

        # among pedal-down notes: is the predicted end closer to key release or to sound end?
        closer_to_key = np.abs(err[down]) < np.abs(err_ped[down])

        n_frames = int(np.ceil(max(r_foff.max(), e_off.max() if len(e_off) else 0) * FRAME_RATE)) + 1
        pp, pr, pf = prf(pedal_mask(ct, cv, n_frames), predicted_pedal(est_path, n_frames))

        m = meta.get(rid, {})
        up_s, dn_s = summarize(err[~down], dur[~down]), summarize(err[down], dur[down])
        row = {
            "record_time": rid, "session": rid[:7], "composer": m.get("composer"),
            "piece": m.get("piece"), "performer": m.get("P1_name"), "skill": m.get("P1_skill"),
            "matched_notes": int(len(match)),
            "share_released_pedal_down": r3(np.mean(down)) if len(down) else None,
            "share_released_half_pedal": r3(np.mean(half)) if len(half) else None,
            "cc64_distinct_values": int(len(np.unique(cv))),
            "all_median_err_ms": round(float(np.median(err)) * 1000, 1) if len(err) else None,
        }
        for prefix, s in (("up", up_s), ("down", dn_s)):
            for k, v in s.items():
                row[f"{prefix}_{k}"] = v
        row["down_share_closer_to_key_release"] = r3(np.mean(closer_to_key)) if down.any() else None
        row["pedal_frame_precision"], row["pedal_frame_recall"], row["pedal_frame_f1"] = r3(pp), r3(pr), r3(pf)
        rows.append(row)
        plot_data.append((rid, m.get("composer"), err[~down], err[down]))

        for k in range(len(ri)):
            note_rows.append({
                "record_time": rid, "pitch": int(r_pitch[ri[k]]), "onset": round(float(r_on[ri[k]]), 4),
                "ref_key_release": round(float(r_koff[ri[k]]), 4), "ref_sound_end": round(float(r_foff[ri[k]]), 4),
                "est_end": round(float(e_off[ei[k]]), 4), "err_vs_key_ms": round(float(err[k]) * 1000, 1),
                "err_vs_sound_end_ms": round(float(err_ped[k]) * 1000, 1),
                "cc64_at_release": int(cc_at_release[k]), "pedal_down": bool(down[k]),
            })

    def write(name, data):
        cols = []
        for d in data:
            cols += [c for c in d if c not in cols]
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(data)

    write("offsets_by_recording.csv", rows)
    write("offsets_notes.csv", note_rows)

    # Console summary
    print(f"\n{'recording':20s} {'performer':10s} {'pedal↓':>6s} {'half':>5s} "
          f"{'up: med ms':>10s} {'ok':>5s} {'down: med ms':>12s} {'ok':>5s} {'late':>5s} "
          f"{'→key':>5s} {'pedalF1':>7s}")
    for x in rows:
        def g(k):
            v = x.get(k)
            return "-" if v is None else v
        print(f"{x['record_time']:20s} {str(x['performer'])[:10]:10s} {g('share_released_pedal_down'):>6} "
              f"{g('share_released_half_pedal'):>5} {g('up_median_ms'):>10} {g('up_share_within_tol'):>5} "
              f"{g('down_median_ms'):>12} {g('down_share_within_tol'):>5} {g('down_share_late'):>5} "
              f"{g('down_share_closer_to_key_release'):>5} {g('pedal_frame_f1'):>7}")
    print("\npedal↓ = share of notes released with pedal down; half = released at 0<CC64<64")
    print("med ms = median signed error (predicted end - key release); ok = share within max(50 ms, 20%)")
    print("late = share ended too late (beyond tolerance); →key = pedal-down notes whose predicted end is")
    print("       closer to key release than to sound end; pedalF1 = predicted vs real pedal, 60 Hz frames")

    # Histograms: one small panel per recording, pedal up vs pedal down
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(plot_data)
    cols_n = 3
    rows_n = int(np.ceil(n / cols_n))
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(4 * cols_n, 2.6 * rows_n), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    bins = np.linspace(-0.5, 1.5, 81)
    for ax, (rid, comp, e_up, e_dn) in zip(axes, plot_data):
        for e, color, label in ((e_up, BLUE, "pedal up at release"), (e_dn, ORANGE, "pedal down at release")):
            if len(e) >= 20:
                ax.hist(np.clip(e, bins[0], bins[-1]), bins=bins, density=True, histtype="step",
                        linewidth=2, color=color, label=f"{label} (n={len(e)})")
        ax.axvline(0, color="#52514e", linewidth=1)
        ax.set_title(f"{comp} · {rid}", fontsize=9, color="#0b0b0b")
        ax.tick_params(labelsize=8, colors="#52514e")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[::cols_n]:
        ax.set_ylabel("density", fontsize=8, color="#52514e")
    for ax in axes[-cols_n:]:
        ax.set_xlabel("predicted end − key release (s)", fontsize=8, color="#52514e")
    fig.suptitle(f"TransKun note-ending error, clean {a.split} audio (positive = ends too late)",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "offset_error_hist.png"), dpi=150, facecolor="#fcfcfb")
    print(f"\nSaved to {out_dir}: offsets_by_recording.csv, offsets_notes.csv, offset_error_hist.png")


if __name__ == "__main__":
    main()
