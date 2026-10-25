#!/usr/bin/env python3
"""
Clean baseline: run TransKun on every recording of a split and score it.

For each recording:
  1. transcribe Audio/<id>.wav  ->  predictions/clean/<split>/<id>.mid
     (skipped if the MIDI already exists, so the script can be stopped and restarted)
  2. score it with evaluate_one.evaluate()  ->  results/clean/<split>/<id>.json
Then it writes one summary table: results/clean/<split>/summary.csv

Usage
  python run_clean.py --split valid
  python run_clean.py --split test     # only once the setup is frozen
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

from evaluate_one import evaluate

SUMMARY_METRICS = [
    ("onset_50ms", "onset"),
    ("onset_100ms", "onset_100"),
    ("onset_offset_key_release", "on+off_key"),
    ("onset_offset_fixed50ms_key_release", "on+off50_key"),
    ("onset_offset_pedal_extended", "on+off_ped"),
    ("onset_velocity", "on+vel"),
    ("frame_key_release", "frame_key"),
    ("frame_pedal_extended", "frame_ped"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/PianoVAM_v1.1")
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    with open(os.path.join(a.data, "manifest.json"), encoding="utf-8") as f:
        recs = [r for r in json.load(f)["recordings"] if r["split"] == a.split]

    pred_dir = os.path.join("predictions", "clean", a.split)
    res_dir = os.path.join("results", "clean", a.split)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    rows = []
    for i, r in enumerate(recs, 1):
        rid = r["record_time"]
        wav = os.path.join(a.data, "Audio", rid + ".wav")
        mid = os.path.join(pred_dir, rid + ".mid")
        tsv = os.path.join(a.data, "TSV", rid + ".tsv")
        print(f"\n[{i}/{len(recs)}] {rid}  {r.get('composer')} - {r.get('piece')}")

        secs = None
        if os.path.exists(mid):
            print("  prediction exists, skipping TransKun")
        else:
            t0 = time.time()
            cmd = [sys.executable, "-m", "transkun.transcribe", wav, mid + ".part.mid", "--device", a.device]
            subprocess.run(cmd, check=True)
            os.replace(mid + ".part.mid", mid)  # only a finished file gets the real name
            secs = round(time.time() - t0, 1)
            print(f"  transcribed in {secs/60:.1f} min")

        res = evaluate(mid, tsv)
        res.update({"split": a.split, "record_time": rid, "composer": r.get("composer"),
                    "piece": r.get("piece"), "audio_dur_s": r["checks"].get("audio_dur_s"),
                    "transcribe_seconds": secs})
        with open(os.path.join(res_dir, rid + ".json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

        row = {"record_time": rid, "composer": r.get("composer"), "piece": r.get("piece"),
               "ref_notes": res["n_ref_notes"], "est_notes": res["n_est_notes"],
               "pedal_presses": res["est_pedal_presses"]}
        for key, short in SUMMARY_METRICS:
            row[short] = res[key]["f1"]
        row["vel_mean_abs_err"] = res.get("raw_velocity_error", {}).get("mean_abs")
        rows.append(row)
        print("  " + "  ".join(f"{short}={row[short]:.3f}" for _, short in SUMMARY_METRICS[:3]))

    # Mean over recordings (each recording counts equally)
    mean = {"record_time": "MEAN", "composer": "", "piece": ""}
    for k in rows[0]:
        if k not in mean:
            vals = [x[k] for x in rows if isinstance(x[k], (int, float))]
            mean[k] = round(sum(vals) / len(vals), 4) if vals else None
    rows.append(mean)

    out = os.path.join(res_dir, "summary.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'recording':20s} " + " ".join(f"{s:>12s}" for _, s in SUMMARY_METRICS))
    for x in rows:
        print(f"{x['record_time']:20s} " + " ".join(f"{x[s]:12.3f}" for _, s in SUMMARY_METRICS))
    print(f"\nF1 scores. Summary saved: {out}")


if __name__ == "__main__":
    main()
