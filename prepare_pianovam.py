#!/usr/bin/env python3
"""
Task 1: fetch the PianoVAM validation and test data at a pinned revision.

What it does
  1. Pins the Hugging Face dataset revision (commit SHA).
  2. Downloads metadata.json at that revision and uses its "split" field
     (the canonical split file since v1.1) to select recordings.
  3. Downloads Audio/, MIDI/, TSV/ for every selected recording that exists,
     plus optional videos for sync checks.
  4. Verifies every file against the checksum Hugging Face stores for it.
  5. Runs sanity checks (WAV format, MIDI notes and CC64, TSV columns, durations).
  6. Writes manifest.json and manifest.csv so the exact data can be cited.

Usage
  pip install "huggingface_hub>=0.24" mido soundfile
  python prepare_pianovam.py --out data/PianoVAM_v1.1 \
      --revision 1f039ab9d6128626bff708faad6472d220b50eaa
  # dry run, nothing large is downloaded:
  python prepare_pianovam.py --list
  # also fetch two videos for synchronization checks:
  python prepare_pianovam.py --video-ids 2024-02-15_20-07-54,2024-09-02_14-10-41
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import sys
from importlib.metadata import version, PackageNotFoundError

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "PianoVAM/PianoVAM_v1"
REPO_TYPE = "dataset"
# Revision checked on 2026-10-04 (lastModified 2026-09-27). Override with --revision.
DEFAULT_REVISION = "1f039ab9d6128626bff708faad6472d220b50eaa"

FOLDERS = {"audio": "Audio", "midi": "MIDI", "tsv": "TSV", "video": "Video"}
TSV_COLUMNS = ["onset", "key_offset", "frame_offset", "note", "velocity"]
DURATION_TOL_S = 2.0  # flag audio/MIDI lengths that differ from metadata by more than this


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/PianoVAM_v1.1")
    p.add_argument("--revision", default=DEFAULT_REVISION,
                   help="Commit SHA to pin. Use 'main' only to discover the latest SHA.")
    p.add_argument("--splits", default="valid,test",
                   help="Values of the metadata.json 'split' field (note: 'valid', not 'validation').")
    p.add_argument("--modalities", default="audio,midi,tsv")
    p.add_argument("--video-ids", default="",
                   help="Comma-separated record_time IDs whose video to fetch for sync checks.")
    p.add_argument("--list", action="store_true", help="Show the plan without downloading media.")
    return p.parse_args()


# ---------- checksums ----------

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(path):
    """Checksum Hugging Face reports for small (non-LFS) files."""
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify(local_path, entry):
    """entry is a RepoFile from list_repo_tree(expand=True)."""
    if entry.lfs is not None:
        expected, got, kind = entry.lfs.sha256, sha256_file(local_path), "sha256"
    else:
        expected, got, kind = entry.blob_id, git_blob_sha1(local_path), "git-sha1"
    return {"checksum_type": kind, "checksum": got, "checksum_ok": got == expected,
            "size_bytes": os.path.getsize(local_path), "size_ok": os.path.getsize(local_path) == entry.size}


# ---------- sanity checks ----------

def parse_meta_duration(s):
    """metadata.json uses strings like '0 days 00:07:46.773.000000'."""
    if not s:
        return None
    m = re.match(r"(\d+) days (\d+):(\d+):(\d+)(?:\.(\d+))?", s)
    if not m:
        return None
    d, h, mi, se, frac = m.groups()
    return int(d) * 86400 + int(h) * 3600 + int(mi) * 60 + int(se) + (int(frac) / 1000 if frac else 0)


def check_audio(path):
    try:
        import soundfile as sf
        info = sf.info(path)
        return {"sr": info.samplerate, "channels": info.channels,
                "audio_dur_s": round(info.frames / info.samplerate, 3), "subtype": info.subtype}
    except ImportError:
        import wave
        with wave.open(path) as w:
            return {"sr": w.getframerate(), "channels": w.getnchannels(),
                    "audio_dur_s": round(w.getnframes() / w.getframerate(), 3), "subtype": f"PCM_{8*w.getsampwidth()}"}


def check_midi(path):
    import mido
    mid = mido.MidiFile(path)
    n_on = n_cc64 = n_cc64_down = 0
    for msg in mid:  # merged tracks, delta times in seconds
        if msg.type == "note_on" and msg.velocity > 0:
            n_on += 1
        elif msg.type == "control_change" and msg.control == 64:
            n_cc64 += 1
            n_cc64_down += msg.value >= 64
    return {"midi_notes": n_on, "midi_cc64_events": n_cc64, "midi_cc64_down": n_cc64_down,
            "midi_dur_s": round(mid.length, 3)}


def check_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f, delimiter="\t") if r]
    header = rows[0] if rows else []
    # Treat the first row as a header if it isn't numeric (names may carry a '#' or differ slightly)
    try:
        float(header[0])
        is_header = False
    except (ValueError, IndexError):
        is_header = True
    has_header = is_header and [h.strip().lstrip("#").strip().lower() for h in header] == TSV_COLUMNS
    body = rows[1:] if is_header else rows
    bad = sum(1 for r in body if len(r) != 5)
    # frame_offset should never be earlier than key_offset (pedal can only extend sound)
    inverted = 0
    for r in body:
        try:
            if float(r[2]) + 1e-6 < float(r[1]):
                inverted += 1
        except (ValueError, IndexError):
            pass
    return {"tsv_rows": len(body), "tsv_header_ok": has_header,
            "tsv_header": "|".join(header) if is_header else "", "tsv_bad_rows": bad,
            "tsv_frame_before_key_offset": inverted}


# ---------- main ----------

def pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    a = parse_args()
    api = HfApi()

    # 1. Pin revision
    info = api.dataset_info(REPO_ID, revision=a.revision)
    sha = info.sha
    if a.revision != sha:
        print(f"WARNING: '{a.revision}' resolved to {sha}. Re-run with --revision {sha} to pin it.")
    print(f"Dataset {REPO_ID} @ {sha} (lastModified {info.last_modified})")

    os.makedirs(a.out, exist_ok=True)

    # 2. Canonical splits from metadata.json
    meta_path = hf_hub_download(REPO_ID, "metadata.json", repo_type=REPO_TYPE, revision=sha, local_dir=a.out)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    rows = list(meta.values()) if isinstance(meta, dict) else meta
    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    all_splits = sorted({r.get("split") for r in rows})
    unknown = [s for s in splits if s not in all_splits]
    if unknown:
        sys.exit(f"Unknown split(s) {unknown}. metadata.json has: {all_splits}")
    selected = [r for r in rows if r.get("split") in splits]
    print("Split counts in metadata.json:",
          {s: sum(r.get('split') == s for r in rows) for s in all_splits})

    # 3. Index the repo folders once, keyed by file stem (= record_time)
    modalities = [m.strip() for m in a.modalities.split(",") if m.strip()]
    video_ids = [v.strip() for v in a.video_ids.split(",") if v.strip()]
    needed = set(modalities) | ({"video"} if video_ids else set())
    index = {}
    for mod in needed:
        index[mod] = {}
        for e in api.list_repo_tree(REPO_ID, path_in_repo=FOLDERS[mod], repo_type=REPO_TYPE,
                                    revision=sha, expand=True):
            if hasattr(e, "size"):  # files only
                stem = os.path.splitext(os.path.basename(e.path))[0]
                index[mod][stem] = e

    # 4. Plan
    plan, records = [], []
    for r in sorted(selected, key=lambda r: (r["split"], r["record_time"])):
        rid = r["record_time"]
        rec = {"record_time": rid, "split": r["split"], "composer": r.get("composer"),
               "piece": r.get("piece"), "performance_method": r.get("performance_method"),
               "meta_dur_s": parse_meta_duration(r.get("duration")), "files": {}, "missing": []}
        mods = modalities + (["video"] if rid in video_ids else [])
        for mod in mods:
            e = index[mod].get(rid)
            if e is None:
                rec["missing"].append(mod)
            else:
                plan.append((rec, mod, e))
        records.append(rec)

    total_gb = sum(e.size for _, _, e in plan) / 1e9
    print(f"{len(records)} recordings selected, {len(plan)} files, {total_gb:.2f} GB")
    for rec in records:
        if rec["missing"]:
            print(f"  MISSING {rec['split']:5s} {rec['record_time']}: {', '.join(rec['missing'])}")
    if a.list:
        for rec, mod, e in plan:
            print(f"  {rec['split']:5s} {mod:5s} {e.path}  {e.size/1e6:.1f} MB")
        return

    # 5. Download + verify
    for i, (rec, mod, e) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {e.path}")
        local = hf_hub_download(REPO_ID, e.path, repo_type=REPO_TYPE, revision=sha, local_dir=a.out)
        v = verify(local, e)
        v["path"] = os.path.relpath(local, a.out)
        rec["files"][mod] = v
        if not (v["checksum_ok"] and v["size_ok"]):
            print(f"  CHECKSUM MISMATCH: {e.path}")

    # 6. Sanity checks
    for rec in records:
        c = {}
        f = rec["files"]
        if "audio" in f:
            c.update(check_audio(os.path.join(a.out, f["audio"]["path"])))
        if "midi" in f:
            c.update(check_midi(os.path.join(a.out, f["midi"]["path"])))
        if "tsv" in f:
            c.update(check_tsv(os.path.join(a.out, f["tsv"]["path"])))
        flags = []
        if c.get("sr") not in (None, 44100):
            flags.append(f"sr={c['sr']}")
        notes = []  # informational, not problems
        md = rec["meta_dur_s"]
        if md and c.get("audio_dur_s") is not None and abs(c["audio_dur_s"] - md) > DURATION_TOL_S:
            flags.append(f"audio length differs from metadata by {c['audio_dur_s']-md:+.1f}s")
        # The MIDI file ends at its last event, so it is normally a few seconds shorter than the
        # recording. Only a MIDI that runs past the end of the audio points to misalignment.
        if c.get("audio_dur_s") is not None and c.get("midi_dur_s") is not None:
            c["midi_end_before_audio_end_s"] = round(c["audio_dur_s"] - c["midi_dur_s"], 3)
            if c["midi_dur_s"] > c["audio_dur_s"] + DURATION_TOL_S:
                flags.append(f"MIDI runs {c['midi_dur_s']-c['audio_dur_s']:.1f}s past the audio end")
        if "midi_notes" in c and "tsv_rows" in c and c["midi_notes"] != c["tsv_rows"]:
            flags.append(f"MIDI notes {c['midi_notes']} != TSV rows {c['tsv_rows']}")
        if c.get("midi_cc64_events") == 0:
            notes.append("no sustain pedal (CC64) used in this performance")
        if c.get("tsv_header") and not c.get("tsv_header_ok"):
            notes.append(f"TSV header is '{c['tsv_header']}'")
        rec["notes"] = notes
        if c.get("tsv_frame_before_key_offset"):
            flags.append(f"{c['tsv_frame_before_key_offset']} notes with frame_offset < key_offset")
        if rec["missing"]:
            flags.append("missing: " + ",".join(rec["missing"]))
        rec["checks"], rec["flags"] = c, flags

    # 7. Manifest
    manifest = {
        "dataset": REPO_ID, "revision": sha, "last_modified": str(info.last_modified),
        "split_source": "metadata.json field 'split'", "splits": splits,
        "modalities": modalities, "video_ids": video_ids,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "huggingface_hub": pkg_version("huggingface_hub"),
                        "mido": pkg_version("mido"), "soundfile": pkg_version("soundfile")},
        "metadata_json": verify(meta_path, next(iter(api.get_paths_info(
            REPO_ID, ["metadata.json"], repo_type=REPO_TYPE, revision=sha, expand=True)))),
        "counts": {s: sum(r["split"] == s for r in records) for s in splits},
        "recordings": records,
    }
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    cols = ["split", "record_time", "composer", "piece", "meta_dur_s", "audio_dur_s", "midi_dur_s",
            "sr", "channels", "midi_notes", "tsv_rows", "midi_cc64_events", "midi_cc64_down",
            "midi_end_before_audio_end_s", "tsv_frame_before_key_offset", "all_checksums_ok",
            "flags", "notes"]
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            row = {k: r.get(k) for k in ("split", "record_time", "composer", "piece", "meta_dur_s")}
            row.update({k: r["checks"].get(k) for k in cols if k in r["checks"]})
            row["all_checksums_ok"] = all(v["checksum_ok"] and v["size_ok"] for v in r["files"].values())
            row["flags"] = "; ".join(r["flags"])
            row["notes"] = "; ".join(r["notes"])
            w.writerow(row)

    bad = [r["record_time"] for r in records if r["flags"]]
    print(f"\nDone. counts={manifest['counts']}  flagged={len(bad)}")
    for r in records:
        if r["flags"]:
            print(f"  PROBLEM {r['split']:5s} {r['record_time']}: {'; '.join(r['flags'])}")
    for r in records:
        if r["notes"]:
            print(f"  note    {r['split']:5s} {r['record_time']}: {'; '.join(r['notes'])}")
    print(f"Manifest: {os.path.join(a.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
