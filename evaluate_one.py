#!/usr/bin/env python3
"""
Score one TransKun transcription against the PianoVAM ground truth.

Usage
  python evaluate_one.py --est test_output.mid ^
      --tsv data\\PianoVAM_v1.1\\TSV\\2024-02-15_20-07-54.tsv ^
      --out results\\clean\\2024-02-15_20-07-54.json

Ground truth comes from the TSV, which holds both note endings:
  key_offset    finger leaves the key  -> primary offset target
  frame_offset  pedal-extended ending  -> reported separately

Metrics follow section 3:
  Onset            pitch + onset within 50 ms (and 100 ms reported separately)
  Onset + offset   offset tolerance max(50 ms, 20% of reference duration)
                   plus a fixed 50 ms offset variant
  Onset + velocity mir_eval velocity metric, tolerance 0.1 (mir_eval fits a
                   linear velocity mapping first); raw velocity error separately
  Frame            60 Hz binary piano rolls; primary roll ends at key release,
                   pedal-extended roll separately
"""

import argparse
import csv
import json
import os
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import pretty_midi
import mir_eval
from mir_eval import transcription as tr
from mir_eval import transcription_velocity as trv

FRAME_RATE = 60  # Hz, shared grid for frame metrics
MIN_DUR = 1e-3   # mir_eval needs positive note durations


def load_reference(tsv_path):
    """Return onsets, key_offsets, frame_offsets, pitches (MIDI), velocities."""
    with open(tsv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f, delimiter="\t") if r]
    try:
        float(rows[0][0])
    except ValueError:
        rows = rows[1:]  # skip header
    a = np.array([[float(x) for x in r[:5]] for r in rows])
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(int), a[:, 4]


def load_estimate(mid_path):
    pm = pretty_midi.PrettyMIDI(mid_path)
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    notes.sort(key=lambda n: (n.start, n.pitch))
    on = np.array([n.start for n in notes])
    off = np.array([n.end for n in notes])
    pitch = np.array([n.pitch for n in notes], dtype=int)
    vel = np.array([n.velocity for n in notes], dtype=float)
    # TransKun writes each predicted pedal press as a CC64 on/off pair, so count the "on" events
    n_pedal = sum(1 for inst in pm.instruments for c in inst.control_changes
                  if c.number == 64 and c.value >= 64)
    return on, off, pitch, vel, n_pedal


def intervals(on, off):
    return np.stack([on, np.maximum(off, on + MIN_DUR)], axis=1)


def prf(p, r, f):
    return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4)}


def piano_roll(on, off, pitch, n_frames):
    roll = np.zeros((128, n_frames), dtype=bool)
    starts = np.ceil(on * FRAME_RATE).astype(int)
    ends = np.maximum(np.ceil(off * FRAME_RATE).astype(int), starts + 1)
    for s, e, p in zip(starts, ends, pitch):
        roll[p, max(s, 0):min(e, n_frames)] = True
    return roll


def frame_prf(ref_roll, est_roll):
    tp = np.logical_and(ref_roll, est_roll).sum()
    p = tp / est_roll.sum() if est_roll.sum() else 0.0
    r = tp / ref_roll.sum() if ref_roll.sum() else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return prf(p, r, f)


def pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def evaluate(est_path, tsv_path):
    """Return a dict with all metrics for one recording."""
    r_on, r_koff, r_foff, r_pitch, r_vel = load_reference(tsv_path)
    e_on, e_off, e_pitch, e_vel, e_pedal = load_estimate(est_path)

    ref_key = intervals(r_on, r_koff)
    ref_ped = intervals(r_on, r_foff)
    est = intervals(e_on, e_off)
    r_hz = mir_eval.util.midi_to_hz(r_pitch)
    e_hz = mir_eval.util.midi_to_hz(e_pitch)

    res = {"est": os.path.basename(est_path), "tsv": os.path.basename(tsv_path),
           "n_ref_notes": int(len(r_on)), "n_est_notes": int(len(e_on)),
           "est_pedal_presses": int(e_pedal)}

    # Onset only
    for tol in (0.05, 0.10):
        p, r, f, _ = tr.precision_recall_f1_overlap(ref_key, r_hz, est, e_hz,
                                                    onset_tolerance=tol, offset_ratio=None)
        res[f"onset_{int(tol*1000)}ms"] = prf(p, r, f)

    # Onset + offset, against key release (primary) and pedal-extended ending
    for name, ref in (("key_release", ref_key), ("pedal_extended", ref_ped)):
        p, r, f, _ = tr.precision_recall_f1_overlap(ref, r_hz, est, e_hz, onset_tolerance=0.05,
                                                    offset_ratio=0.2, offset_min_tolerance=0.05)
        res[f"onset_offset_{name}"] = prf(p, r, f)
        # offset_ratio=0 makes the tolerance a flat max(50 ms, 0) = 50 ms
        p, r, f, _ = tr.precision_recall_f1_overlap(ref, r_hz, est, e_hz, onset_tolerance=0.05,
                                                    offset_ratio=0.0, offset_min_tolerance=0.05)
        res[f"onset_offset_fixed50ms_{name}"] = prf(p, r, f)

    # Onset + velocity (offset ignored)
    p, r, f, _ = trv.precision_recall_f1_overlap(ref_key, r_hz, r_vel, est, e_hz, e_vel,
                                                 onset_tolerance=0.05, offset_ratio=None,
                                                 velocity_tolerance=0.1)
    res["onset_velocity"] = prf(p, r, f)
    res["onset_velocity_note"] = "mir_eval fits a linear mapping from estimated to reference velocity before applying the 0.1 tolerance"

    # Raw velocity error on onset-matched notes (no mapping)
    match = tr.match_notes(ref_key, r_hz, est, e_hz, onset_tolerance=0.05, offset_ratio=None)
    if match:
        ri, ei = zip(*match)
        diff = e_vel[list(ei)] - r_vel[list(ri)]
        res["raw_velocity_error"] = {"matched_notes": len(match),
                                     "mean_abs": round(float(np.mean(np.abs(diff))), 2),
                                     "mean_signed": round(float(np.mean(diff)), 2),
                                     "median_abs": round(float(np.median(np.abs(diff))), 2)}

    # Frame metrics on a shared 60 Hz grid
    n_frames = int(np.ceil(max(r_foff.max(), r_koff.max(), e_off.max() if len(e_off) else 0) * FRAME_RATE)) + 1
    est_roll = piano_roll(e_on, e_off, e_pitch, n_frames)
    res["frame_key_release"] = frame_prf(piano_roll(r_on, r_koff, r_pitch, n_frames), est_roll)
    res["frame_pedal_extended"] = frame_prf(piano_roll(r_on, r_foff, r_pitch, n_frames), est_roll)

    res["versions"] = {"mir_eval": pkg_version("mir_eval"), "transkun": pkg_version("transkun"),
                       "pretty_midi": pkg_version("pretty_midi"), "torch": pkg_version("torch")}
    return res


def print_table(res):
    print(f"\nReference notes: {res['n_ref_notes']}   Estimated notes: {res['n_est_notes']}"
          f"   Pedal presses in TransKun output: {res['est_pedal_presses']}\n")
    print(f"{'Metric':45s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    for k, v in res.items():
        if isinstance(v, dict) and "f1" in v:
            print(f"{k:45s} {v['precision']:7.3f} {v['recall']:7.3f} {v['f1']:7.3f}")
    if "raw_velocity_error" in res:
        rv = res["raw_velocity_error"]
        print(f"\nRaw velocity error (MIDI units, {rv['matched_notes']} matched notes): "
              f"mean |diff| {rv['mean_abs']}, mean signed {rv['mean_signed']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True, help="TransKun output MIDI")
    ap.add_argument("--tsv", required=True, help="PianoVAM TSV ground truth")
    ap.add_argument("--out", help="Optional JSON file for the results")
    a = ap.parse_args()
    res = evaluate(a.est, a.tsv)
    print_table(res)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved: {a.out}")


if __name__ == "__main__":
    main()
