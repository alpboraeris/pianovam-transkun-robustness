#!/usr/bin/env python3
"""
Check that the PianoVAM TSV labels agree with the performance MIDI.

For every recording in the chosen splits it compares, note by note:
  TSV onset       vs MIDI note-on
  TSV key_offset  vs MIDI note-off   (primary offset target)
  TSV note/velocity vs MIDI pitch/velocity
and also reports how much longer frame_offset (pedal-extended) is than key_offset.

Usage
  python check_ground_truth.py --data data\\PianoVAM_v1.1 --splits valid,test
Writes ground_truth_check.csv into the data folder.
"""

import argparse
import csv
import json
import os

import numpy as np
import pretty_midi


def load_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f, delimiter="\t") if r]
    try:
        float(rows[0][0])
    except ValueError:
        rows = rows[1:]
    return np.array([[float(x) for x in r[:5]] for r in rows])  # onset, key_off, frame_off, pitch, vel


def load_midi(path):
    pm = pretty_midi.PrettyMIDI(path)  # note.end = note-off; pretty_midi does not apply the pedal
    notes = [n for inst in pm.instruments for n in inst.notes]
    return np.array([[n.start, n.end, n.pitch, n.velocity] for n in notes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/PianoVAM_v1.1")
    ap.add_argument("--splits", default="valid,test")
    a = ap.parse_args()

    with open(os.path.join(a.data, "manifest.json"), encoding="utf-8") as f:
        recs = [r for r in json.load(f)["recordings"] if r["split"] in a.splits.split(",")]

    out_rows = []
    for r in recs:
        rid = r["record_time"]
        t = load_tsv(os.path.join(a.data, "TSV", rid + ".tsv"))
        m = load_midi(os.path.join(a.data, "MIDI", rid + ".mid"))
        # Sort both by (onset, pitch) so the same note sits in the same row
        t = t[np.lexsort((t[:, 3], np.round(t[:, 0], 4)))]
        m = m[np.lexsort((m[:, 2], np.round(m[:, 0], 4)))]
        row = {"split": r["split"], "record_time": rid, "tsv_notes": len(t), "midi_notes": len(m)}
        if len(t) == len(m):
            row["pitch_mismatches"] = int(np.sum(t[:, 3] != m[:, 2]))
            row["velocity_mismatches"] = int(np.sum(t[:, 4] != m[:, 3]))
            row["max_onset_diff_ms"] = round(float(np.max(np.abs(t[:, 0] - m[:, 0]))) * 1000, 3)
            koff = np.abs(t[:, 1] - m[:, 1])
            row["max_key_offset_diff_ms"] = round(float(koff.max()) * 1000, 3)
            row["notes_key_offset_diff_over_1ms"] = int(np.sum(koff > 0.001))
        ext = t[:, 2] - t[:, 1]
        row["pedal_extension_median_s"] = round(float(np.median(ext)), 3)
        row["pedal_extension_mean_s"] = round(float(np.mean(ext)), 3)
        row["share_notes_extended_over_50ms"] = round(float(np.mean(ext > 0.05)), 3)
        out_rows.append(row)

    cols = list(out_rows[0].keys())
    for r in out_rows:
        cols += [k for k in r if k not in cols]
    out = os.path.join(a.data, "ground_truth_check.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    print(f"{'split':5s} {'recording':20s} {'notes':>6s} {'pitch≠':>6s} {'vel≠':>5s} "
          f"{'max onset Δ ms':>14s} {'max key-off Δ ms':>16s} {'pedal ext. median s':>19s}")
    for r in out_rows:
        print(f"{r['split']:5s} {r['record_time']:20s} {r['tsv_notes']:6d} {r.get('pitch_mismatches','?'):>6} "
              f"{r.get('velocity_mismatches','?'):>5} {r.get('max_onset_diff_ms','?'):>14} "
              f"{r.get('max_key_offset_diff_ms','?'):>16} {r['pedal_extension_median_s']:>19}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
