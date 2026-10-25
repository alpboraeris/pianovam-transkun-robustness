#!/usr/bin/env python3
"""
Controlled noise experiment (section 2)

For each recording and each condition (noise excerpt x SNR):
  1. mix the noise into the original piano audio at the target SNR
  2. save the mixture as 16-bit WAV (TransKun expects 16-bit samples)
  3. transcribe it with TransKun
  4. score it against the ground truth with evaluate_one.evaluate()
  5. delete the mixture WAV again (unless --keep-audio), the MIDI prediction is kept

The script is restartable: conditions whose prediction MIDI already exists are not
transcribed again.

SNR definition (fixed for the whole study)
  SNR_dB = 10 log10( P_piano / P_noise ), P = mean square over the whole recording,
  all channels. The noise (mono) is added identically to every channel. If the
  mixture would clip, the whole mixture (piano + noise) is scaled down, which
  leaves the SNR unchanged; the scale factor is recorded.

Noise
  speech / environment / music  excerpts from noise_bank/ (build_noise_bank.py),
                                looped to the length of the recording
  white                         Gaussian, numpy default_rng(seed), seeds 0, 1, 2

Designs
  --design all     every recording gets all 12 noises (3 per type) x every SNR
  --design first   every recording gets the SAME first excerpt of each type (4 noises)
                   x every SNR; recordings are directly comparable
  --design rotate  every recording gets ONE excerpt per type (4 noises) x every SNR;
                   the excerpt index rotates with the recording index, so each excerpt
                   is used across the split (recording and excerpt are then confounded)

Usage
  # quick check on one recording, one condition (about 4 min on CPU):
  python run_noise.py --split valid --recordings 2024-09-02_12-45-12 --types white --snrs 0 --seeds 0
  # development set, same 4 noises for every recording (about 4.4 h on CPU):
  python run_noise.py --split valid --design first --recordings 2024-09-02_12-45-12,2024-02-15_20-38-23,2024-09-02_19-40-55
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import soundfile as sf

from evaluate_one import evaluate

SR = 44100
CHUNK = SR * 30
SNRS = [20, 10, 0, -10, -20]
TYPES = ["speech", "environment", "music", "white"]
WHITE_SEEDS = [0, 1, 2]
PEAK_LIMIT = 0.999

SUMMARY_KEYS = [
    ("onset_50ms", "onset"), ("onset_100ms", "onset_100"),
    ("onset_offset_key_release", "on+off_key"), ("onset_offset_fixed50ms_key_release", "on+off50_key"),
    ("onset_offset_pedal_extended", "on+off_ped"), ("onset_velocity", "on+vel"),
    ("frame_key_release", "frame_key"), ("frame_pedal_extended", "frame_ped"),
]


# ---------- noise sources ----------

class LoopedExcerpt:
    """Noise excerpt repeated end-to-end; read(start, n) returns samples [start, start+n)."""
    def __init__(self, path):
        x, sr = sf.read(path, dtype="float32")
        assert sr == SR and x.ndim == 1, f"{path}: expected mono {SR} Hz"
        self.x = x

    def read(self, start, n):
        idx = (np.arange(start, start + n) % len(self.x))
        return self.x[idx]


class WhiteNoise:
    """Gaussian white noise, reproducible from the seed, read chunk by chunk in order."""
    def __init__(self, seed):
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.pos = 0

    def read(self, start, n):
        assert start == self.pos, "white noise must be read sequentially"
        self.pos += n
        return self.rng.standard_normal(n).astype(np.float32)


def make_source(cond, bank_dir):
    if cond["type"] == "white":
        return WhiteNoise(cond["seed"])
    return LoopedExcerpt(os.path.join(bank_dir, cond["file"]))


def reset(src):
    if isinstance(src, WhiteNoise):
        src.reset()


# ---------- mixing ----------

def mean_power_piano(path):
    total, count = 0.0, 0
    with sf.SoundFile(path) as f:
        for block in f.blocks(blocksize=CHUNK, dtype="float32", always_2d=True):
            total += float(np.sum(block.astype(np.float64) ** 2))
            count += block.size
    return total / count


def mean_power_noise(src, n_frames):
    reset(src)
    total = 0.0
    for start in range(0, n_frames, CHUNK):
        n = min(CHUNK, n_frames - start)
        total += float(np.sum(src.read(start, n).astype(np.float64) ** 2))
    return total / n_frames


def mix_to_wav(piano_path, src, snr_db, out_path):
    """Write piano + scaled noise as 16-bit WAV. Returns a dict describing the mix."""
    info = sf.info(piano_path)
    assert info.samplerate == SR, f"{piano_path}: {info.samplerate} Hz, expected {SR}"
    n = info.frames
    p_piano = mean_power_piano(piano_path)
    p_noise = mean_power_noise(src, n)
    gain = np.sqrt(p_piano / (p_noise * 10 ** (snr_db / 10)))

    # pass 1: peak of the mixture, so clipping can be avoided by one global scale
    peak = 0.0
    reset(src)
    with sf.SoundFile(piano_path) as f:
        start = 0
        for block in f.blocks(blocksize=CHUNK, dtype="float32", always_2d=True):
            m = block + gain * src.read(start, len(block))[:, None]
            peak = max(peak, float(np.max(np.abs(m))))
            start += len(block)
    scale = min(1.0, PEAK_LIMIT / peak) if peak > 0 else 1.0

    # pass 2: write, and measure the achieved SNR as a check
    sig_e, noi_e = 0.0, 0.0
    reset(src)
    with sf.SoundFile(piano_path) as f, \
            sf.SoundFile(out_path, "w", samplerate=SR, channels=info.channels, subtype="PCM_16") as out:
        start = 0
        for block in f.blocks(blocksize=CHUNK, dtype="float32", always_2d=True):
            nz = gain * src.read(start, len(block))[:, None] * np.ones((1, info.channels), np.float32)
            sig_e += float(np.sum(block.astype(np.float64) ** 2))
            noi_e += float(np.sum(nz.astype(np.float64) ** 2))
            out.write(((block + nz) * scale).astype(np.float32))
            start += len(block)
    return {"noise_gain": float(gain), "mix_scale": float(scale), "peak_before_scale": peak,
            "snr_check_db": round(10 * np.log10(sig_e / noi_e), 3), "channels": info.channels,
            "frames": n}


# ---------- conditions ----------

def all_conditions(bank, types, seeds):
    conds = []
    for t in types:
        if t == "white":
            for s in seeds:
                conds.append({"type": "white", "noise_id": f"white_seed{s}", "seed": s})
        else:
            for e in [e for e in bank["excerpts"] if e["category"] == t]:
                conds.append({"type": t, "noise_id": e["id"], "file": e["file"]})
    return conds


def conditions_for(rec_index, design, conds):
    if design == "all":
        return conds
    out = []  # one excerpt per type
    for t in TYPES:
        group = [c for c in conds if c["type"] == t]
        if group:
            out.append(group[0] if design == "first" else group[rec_index % len(group)])
    return out


def snr_tag(snr):
    return f"snr{snr:+03d}"  # snr+20, snr+00, snr-10


def pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/PianoVAM_v1.1")
    ap.add_argument("--bank", default="noise_bank")
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--recordings", default="", help="comma-separated record_time IDs (default: whole split)")
    ap.add_argument("--design", default="all", choices=["all", "first", "rotate"])
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--snrs", default=",".join(str(s) for s in SNRS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in WHITE_SEEDS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--keep-audio", action="store_true", help="keep the noisy WAV files (large)")
    a = ap.parse_args()

    with open(os.path.join(a.bank, "noise_bank.json"), encoding="utf-8") as f:
        bank = json.load(f)
    with open(os.path.join(a.data, "manifest.json"), encoding="utf-8") as f:
        recs = sorted((r for r in json.load(f)["recordings"] if r["split"] == a.split),
                      key=lambda r: r["record_time"])
    if a.recordings:
        wanted = [x.strip() for x in a.recordings.split(",")]
        missing = [w for w in wanted if w not in {r["record_time"] for r in recs}]
        if missing:
            sys.exit(f"Not in split {a.split}: {missing}")
        recs = [r for r in recs if r["record_time"] in wanted]
    # rotation index = position among the recordings of THIS run (sorted by ID),
    # so any selection of n recordings uses the excerpts as evenly as possible
    rec_index = {r["record_time"]: i for i, r in enumerate(recs)}

    types = [t.strip() for t in a.types.split(",")]
    snrs = [int(s) for s in a.snrs.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    conds = all_conditions(bank, types, seeds)

    base_pred = os.path.join("predictions", "noise", a.split)
    base_res = os.path.join("results", "noise", a.split)
    tmp_dir = os.path.join("tmp_mixtures")
    os.makedirs(base_res, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # record the settings of this run
    run_cfg = {"split": a.split, "design": a.design, "snrs_db": snrs, "types": types,
               "white_seeds": seeds, "snr_definition": "10*log10(P_piano/P_noise), mean square over whole "
               "recording and all channels; mono noise added to every channel; global down-scaling if peak > "
               f"{PEAK_LIMIT}", "noise_bank_speakers": bank.get("speakers"),
               "noise_bank_sha256": {e["id"]: e["sha256"] for e in bank["excerpts"]},
               "versions": {"transkun": pkg_version("transkun"), "torch": pkg_version("torch"),
                            "mir_eval": pkg_version("mir_eval"), "numpy": pkg_version("numpy"),
                            "soundfile": pkg_version("soundfile")},
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(base_res, f"run_config_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    jobs = [(r, c, s) for r in recs for c in conditions_for(rec_index[r["record_time"]], a.design, conds)
            for s in snrs]
    print(f"{len(recs)} recording(s), {len(jobs)} noisy conditions in total")

    for k, (r, c, snr) in enumerate(jobs, 1):
        rid = r["record_time"]
        name = f"{c['noise_id']}_{snr_tag(snr)}"
        pred = os.path.join(base_pred, rid, name + ".mid")
        res_path = os.path.join(base_res, rid, name + ".json")
        os.makedirs(os.path.dirname(pred), exist_ok=True)
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        print(f"\n[{k}/{len(jobs)}] {rid}  {c['noise_id']}  SNR {snr:+d} dB")

        mix_info, secs = None, None
        if os.path.exists(pred):
            print("  prediction exists, skipping mixing and TransKun")
            if os.path.exists(res_path):
                continue
        else:
            wav = os.path.join(tmp_dir, f"{rid}_{name}.wav")
            mix_info = mix_to_wav(os.path.join(a.data, "Audio", rid + ".wav"), make_source(c, a.bank), snr, wav)
            print(f"  mixed: achieved SNR {mix_info['snr_check_db']:+.2f} dB, scale {mix_info['mix_scale']:.3f}")
            t0 = time.time()
            subprocess.run([sys.executable, "-m", "transkun.transcribe", wav, pred + ".part.mid",
                            "--device", a.device], check=True)
            os.replace(pred + ".part.mid", pred)
            secs = round(time.time() - t0, 1)
            print(f"  transcribed in {secs/60:.1f} min")
            if not a.keep_audio:
                os.remove(wav)

        res = evaluate(pred, os.path.join(a.data, "TSV", rid + ".tsv"))
        res.update({"split": a.split, "record_time": rid, "noise_type": c["type"], "noise_id": c["noise_id"],
                    "snr_db": snr, "design": a.design, "mix": mix_info, "transcribe_seconds": secs})
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print("  " + "  ".join(f"{short}={res[key]['f1']:.3f}" for key, short in SUMMARY_KEYS[:3]))

    # rebuild the long-format table from every result JSON of this split (clean included)
    rows = []
    for jp in sorted(glob.glob(os.path.join(base_res, "*", "*.json"))):
        with open(jp, encoding="utf-8") as f:
            x = json.load(f)
        rows.append(row_from(x, x["noise_type"], x["noise_id"], x["snr_db"]))
    for jp in sorted(glob.glob(os.path.join("results", "clean", a.split, "*.json"))):
        with open(jp, encoding="utf-8") as f:
            x = json.load(f)
        rows.append(row_from(x, "clean", "clean", None))
    out = os.path.join(base_res, "noise_results.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nAll results ({len(rows)} rows, clean included): {out}")


def row_from(x, ntype, nid, snr):
    row = {"record_time": x.get("record_time") or x["tsv"].replace(".tsv", ""), "noise_type": ntype,
           "noise_id": nid, "snr_db": snr, "ref_notes": x["n_ref_notes"], "est_notes": x["n_est_notes"],
           "pedal_presses": x.get("est_pedal_presses")}
    for key, short in SUMMARY_KEYS:
        row[short] = x[key]["f1"]
        if key in ("onset_50ms", "onset_offset_key_release"):
            row[short + "_P"] = x[key]["precision"]
            row[short + "_R"] = x[key]["recall"]
    row["vel_mean_abs_err"] = x.get("raw_velocity_error", {}).get("mean_abs")
    return row


if __name__ == "__main__":
    main()
