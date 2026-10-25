#!/usr/bin/env python3
"""
Build the frozen noise bank: 9 noise excerpts (3 speech, 3 environment, 3 music).

Run this ONCE. Its output (noise_bank/ + noise_bank.json) is the fixed set of noises
used for every noisy condition, on validation and later on test.

Sources (all freely licensed, see noise_bank.json for the full record):
  speech       LibriSpeech dev-clean (CC BY 4.0): 3 speakers, utterances of each
               speaker concatenated into one excerpt. No background music.
  environment  DEMAND (Thiemann et al. 2013): microphone ch01 of
               PCAFETER (cafeteria), STRAFFIC (street traffic), DWASHING (washing machine)
  music        3 non-piano recordings from the Internet Archive (public domain):
               string quartet, symphony orchestra, military brass band
  white noise  not stored; generated in run_noise.py from fixed seeds 0, 1, 2

Processing of every excerpt
  - converted to mono (channel mean) and resampled to 44.1 kHz (soxr, as TransKun uses)
  - leading/trailing near-silence trimmed (music only)
  - cut to at most --seconds (default 300 s)
  - loop seam smoothed with a 0.5 s equal-power crossfade, so looping it to cover a
    long piano recording creates no click
  - scaled to RMS 0.1 and saved as 32-bit float WAV (no clipping, no quantisation)

Usage
  python build_noise_bank.py --sources noise_sources --out noise_bank
"""

import argparse
import glob
import hashlib
import json
import os
import platform
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import soundfile as sf
import soxr

SR = 44100
XFADE_S = 0.5
TARGET_RMS = 0.1
SPEAKER_SEED = 0  # fixed seed for choosing the LibriSpeech speakers

ENVIRONMENTS = [
    ("env_cafeteria", "DEMAND/PCAFETER/ch01.wav", "PCAFETER: university cafeteria"),
    ("env_street", "DEMAND/STRAFFIC/ch01.wav", "STRAFFIC: busy street with traffic"),
    ("env_washing", "DEMAND/DWASHING/ch01.wav", "DWASHING: washing machine in a laundry room"),
]
DEMAND_CITATION = ("J. Thiemann, N. Ito, E. Vincent, 'The Diverse Environments Multi-channel Acoustic "
                   "Noise Database (DEMAND)', Proc. Meetings on Acoustics 19, 2013. "
                   "https://zenodo.org/records/1227121")

MUSIC = [
    ("music_strings",
     "*Borodin*",
     "Borodin, String Quartet No. 1, I. Moderato - Allegro; Musopen String Quartet (string quartet)",
     "Public Domain Mark 1.0",
     "https://archive.org/details/MusopenCollectionAsFlac"),
    ("music_orchestra",
     "*Eroica*",
     "Beethoven, Symphony No. 3 'Eroica', I. Allegro con brio; Musopen recording (symphony orchestra)",
     "Public Domain Mark 1.0",
     "https://archive.org/details/MusopenCollectionAsFlac"),
    ("music_brassband",
     "*tars*tripes*",
     "Sousa, The Stars and Stripes Forever; U.S. Marine Corps Band (brass band with percussion)",
     "Public domain (creativecommons.org/licenses/publicdomain)",
     "https://archive.org/details/StarsAndStripesForever"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_mono_44k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = soxr.resample(x, sr, SR).astype(np.float32)
    return x, sr


def trim_silence(x, rel_db=-50.0, win=2048):
    """Cut leading/trailing parts quieter than rel_db below the loudest window."""
    n = len(x) // win
    if n < 2:
        return x
    e = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1) + 1e-20)
    keep = np.where(20 * np.log10(e / e.max()) > rel_db)[0]
    return x[keep[0] * win:(keep[-1] + 1) * win]


def make_loopable(x):
    """Equal-power crossfade of the tail into the head, so x can be repeated seamlessly."""
    L = int(XFADE_S * SR)
    if len(x) < 4 * L:
        return x
    t = np.linspace(0, np.pi / 2, L, dtype=np.float32)
    y = x[: len(x) - L].copy()
    y[:L] = x[:L] * np.sin(t) + x[len(x) - L:] * np.cos(t)
    return y


def finish(x, seconds):
    if len(x) < 10 * SR:
        raise SystemExit(f"Excerpt shorter than 10 s ({len(x)/SR:.1f} s): source file empty or broken?")
    x = x[: int(seconds * SR)]
    x = make_loopable(x)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return (x * (TARGET_RMS / rms)).astype(np.float32), rms


def read_speakers(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(";") or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5 and parts[2] == "dev-clean":
                out[parts[0]] = {"sex": parts[1], "name": parts[4]}
    return out


def pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="noise_sources")
    ap.add_argument("--out", default="noise_bank")
    ap.add_argument("--seconds", type=float, default=300.0, help="max excerpt length")
    ap.add_argument("--speakers", default="",
                    help="optional comma-separated LibriSpeech speaker IDs; default: 2 female + 1 male, chosen with a fixed seed")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bank = []

    def save(eid, x, entry):
        path = os.path.join(a.out, eid + ".wav")
        sf.write(path, x, SR, subtype="FLOAT")
        entry.update({"id": eid, "file": os.path.basename(path), "duration_s": round(len(x) / SR, 2),
                      "sha256": sha256(path)})
        bank.append(entry)
        print(f"  {eid:18s} {entry['duration_s']:7.1f} s  {entry['description']}")

    # ---- speech: LibriSpeech dev-clean ----
    libri = os.path.join(a.sources, "LibriSpeech")
    spk_info = read_speakers(os.path.join(libri, "SPEAKERS.TXT"))
    available = sorted(d for d in os.listdir(os.path.join(libri, "dev-clean")) if d.isdigit())
    if a.speakers:
        chosen = [s.strip() for s in a.speakers.split(",")]
    else:
        rng = np.random.default_rng(SPEAKER_SEED)
        fem = [s for s in available if spk_info.get(s, {}).get("sex") == "F"]
        male = [s for s in available if spk_info.get(s, {}).get("sex") == "M"]
        chosen = sorted(list(rng.choice(fem, 2, replace=False)) + list(rng.choice(male, 1, replace=False)))
    for spk in chosen:
        files = sorted(glob.glob(os.path.join(libri, "dev-clean", spk, "*", "*.flac")))
        parts, total = [], 0
        for fp in files:
            x, src_sr = load_mono_44k(fp)
            parts.append(x)
            total += len(x)
            if total >= a.seconds * SR + XFADE_S * SR:
                break
        x, src_rms = finish(np.concatenate(parts), a.seconds)
        info = spk_info.get(spk, {})
        save(f"speech_{spk}", x, {
            "category": "speech",
            "description": f"LibriSpeech dev-clean speaker {spk} ({info.get('sex', '?')}), "
                           f"{len(parts)} utterances concatenated",
            "source": f"LibriSpeech/dev-clean/{spk}/ (first {len(parts)} files in sorted order)",
            "source_sample_rate": src_sr, "license": "CC BY 4.0",
            "citation": "V. Panayotov et al., 'LibriSpeech: an ASR corpus based on public domain audio "
                        "books', ICASSP 2015. https://www.openslr.org/12/",
        })

    # ---- environment: DEMAND ----
    for eid, rel, desc in ENVIRONMENTS:
        fp = os.path.join(a.sources, rel)
        x, src_sr = load_mono_44k(fp)
        x, _ = finish(x, a.seconds)
        save(eid, x, {"category": "environment", "description": desc, "source": rel,
                      "source_sha256": sha256(fp), "source_sample_rate": src_sr,
                      "license": "see Zenodo record (CC BY 4.0 listed; PDF states CC BY-SA 3.0)",
                      "citation": DEMAND_CITATION})

    # ---- music ----
    music_dir = os.path.join(a.sources, "music")
    for eid, pattern, desc, lic, url in MUSIC:
        hits = [p for p in glob.glob(os.path.join(music_dir, pattern))
                if os.path.splitext(p)[1].lower() in (".flac", ".ogg", ".wav", ".mp3")]
        if len(hits) != 1:
            listing = "\n    ".join(os.listdir(music_dir)) if os.path.isdir(music_dir) else "(folder missing)"
            raise SystemExit(f"\nMusic file for {eid} not found uniquely (pattern '{pattern}', "
                             f"{len(hits)} matches). Files in {music_dir}:\n    {listing}")
        fp = hits[0]
        if os.path.getsize(fp) < 200_000:
            raise SystemExit(f"\n{fp} is only {os.path.getsize(fp)} bytes: download incomplete? Download it again.")
        fname = os.path.basename(fp)
        try:
            x, src_sr = load_mono_44k(fp)
        except Exception as e:
            raise SystemExit(f"\nCould not read {fp}: {e}\nThe file may be damaged; download it again.")
        x, _ = finish(trim_silence(x), a.seconds)
        save(eid, x, {"category": "music", "description": desc, "source": f"music/{fname}",
                      "source_sha256": sha256(fp), "source_sample_rate": src_sr,
                      "license": lic, "citation": url})

    meta = {
        "sample_rate": SR, "max_seconds": a.seconds, "crossfade_s": XFADE_S, "rms": TARGET_RMS,
        "format": "mono 32-bit float WAV", "speaker_seed": SPEAKER_SEED, "speakers": chosen,
        "white_noise": {"seeds": [0, 1, 2], "note": "generated in run_noise.py, Gaussian, not stored"},
        "versions": {"python": platform.python_version(), "numpy": pkg_version("numpy"),
                     "soundfile": pkg_version("soundfile"), "soxr": pkg_version("soxr")},
        "excerpts": bank,
    }
    with open(os.path.join(a.out, "noise_bank.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n{len(bank)} excerpts written to {a.out}/ ; record: {a.out}/noise_bank.json")
    print("Listen to a few of them before continuing.")


if __name__ == "__main__":
    main()
