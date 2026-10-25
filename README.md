# TransKun on PianoVAM: robustness to noise and the sustain pedal

This repository evaluates the TransKun piano transcription model on the PianoVAM dataset. It answers three questions:

1. How well does TransKun transcribe clean real piano recordings?
2. How does it degrade under additive noise (speech, environment, other music, white noise) at controlled SNR levels?
3. Which notes does it miss, and how does the sustain pedal affect its note endings?

**Status:** all results are on the **validation split**. The setup is frozen there and will be run unchanged on the test split.

## Pipeline

Run in this order. Every step is restartable and skips work that already exists.

| Step | Command | Output |
|---|---|---|
| 1. Download data | `python prepare_pianovam.py --out data/PianoVAM_v1.1` | `data/PianoVAM_v1.1/` with `manifest.csv` |
| 2. Check ground truth | `python check_ground_truth.py` | `ground_truth_check.csv` |
| 3. Clean baseline | `python run_clean.py --split valid` | `results/clean/valid/` |
| 4. Build noise bank | `python build_noise_bank.py` | `noise_bank/` |
| 5. Noise experiment | `python run_noise.py --split valid --design first --recordings 2024-02-15_20-38-23,2024-09-02_12-45-12,2024-09-02_19-40-55` | `results/noise/valid/` |
| 6. Noise plots | `python plot_noise.py --split valid` | `results/noise/valid/plots/` |
| 7. Error categories | `python analyze_errors.py --split valid` | `results/analysis/errors/valid/` |
| 8. Note endings and pedal | `python analyze_offsets.py --split valid` | `results/analysis/valid/` |

`evaluate_one.py` holds the scoring used by steps 3 and 5.

## Data

- **PianoVAM v1.1** from Hugging Face (`PianoVAM/PianoVAM_v1`), pinned to commit `1f039ab9d6128626bff708faad6472d220b50eaa`.
- Splits are read from `metadata.json`: **9 valid** and **9 test** recordings, each with audio (44.1 kHz), performance MIDI with sustain pedal (CC64) and TSV annotations.
- All files are checksum-verified. TSV and MIDI agree note by note for all 18 recordings (`ground_truth_check.csv`).
- TSV columns: `onset, key_offset, frame_offset, note, velocity`. `key_offset` is the physical key release (primary note end); `frame_offset` is the pedal-extended end.

## Model

TransKun 2.0.1 with its bundled checkpoint `pretrained/2.0.pt` (sha256 `50a80010effc2a59ffcd068a95cd2b29bd7f23a27a3515bc3ccd209c89a3d44c`). It predicts notes and the sustain pedal as separate outputs; notes are not extended by the pedal.

## Metrics

All scores are computed against the TSV ground truth with `mir_eval` 0.8.2.

| Metric | A predicted note is correct if… |
|---|---|
| Onset (±50 ms, ±100 ms) | pitch matches and onset is within the tolerance |
| Onset + offset | additionally, the end is within max(50 ms, 20% of note duration); also a fixed 50 ms variant |
| Onset + velocity | onset match and velocity within 0.1 on a 0–1 scale (mir_eval fits a linear velocity mapping first) |
| Frame | precision/recall on binary piano rolls at 60 Hz |

Offset and frame metrics are reported against the key release (primary) and the pedal-extended end (separately). The raw velocity error (mean absolute difference in MIDI units, no mapping) is reported separately. The frame metric is implemented directly and matches `mir_eval.multipitch` once duplicate pitches are removed.

## Noise experiment

- **Noise types:** three identified excerpts each, stored with source, licence and sha256 in `noise_bank/noise_bank.json`.
  - Speech: LibriSpeech dev-clean, speakers 251, 5338, 6345 (CC BY 4.0)
  - Environment: DEMAND PCAFETER, STRAFFIC, DWASHING, microphone ch01
  - Other music: Borodin String Quartet No. 1, Beethoven Symphony No. 3, Sousa The Stars and Stripes Forever (public domain)
  - White noise: Gaussian, seeds 0, 1, 2
- **SNR levels:** +20, +10, 0, −10, −20 dB, with SNR = 10·log10(P_piano / P_noise) over the whole recording.
- **Mixing:** excerpts are mono, 44.1 kHz, at most 5 min, looped to the piece length with a 0.5 s crossfade. The same noise is added to both stereo channels. If a mix would clip, the whole mix is scaled down, which leaves the SNR unchanged. Mixtures are written as 16-bit WAV and deleted after transcription.
- **Validation design:** 3 recordings (heavy pedal, no pedal, moderate pedal) × the first excerpt of each type × 5 SNR levels = 60 runs. Every recording receives identical noise, so recordings are directly comparable.

## Analyses

**Error categories** (`analyze_errors.py`). Notes are matched on pitch and onset (±50 ms). Recall is reported by octave, register (low A0–B2, mid C3–B4, high C5–C8), black/white key (also within register), velocity and polyphony. Each missed note gets one reason, checked in this order:

| Reason | A predicted note exists… |
|---|---|
| timing | at the same pitch, 50–200 ms away |
| octave | 12 or 24 semitones away, within 50 ms |
| neighbour | 1 or 2 semitones away, within 50 ms |
| undetected | none of the above |

False positives are classified the same way, with "harmonic" (12, 19 or 24 semitones) instead of "octave" and "spurious" when nothing is near. The thresholds beyond 50 ms are heuristic choices.

**Note endings and pedal** (`analyze_offsets.py`). For every correctly found note: signed error = predicted end − key release, split by whether the pedal was down (CC64 ≥ 64) at the key release. Pedal-up and pedal-down notes are compared within the same recording. Also reported: TransKun's pedal detection (frame F1 at 60 Hz).

## Main results (validation)

Clean audio, mean over 9 recordings:

| Onset | Onset+offset (key release) | Onset+offset (pedal-extended) | Onset+velocity | Frame (key release) |
|---|---|---|---|---|
| 0.989 | 0.654 | 0.247 | 0.947 | 0.747 |

Onset F1 under noise, mean over 3 recordings (clean 0.990):

| Noise | +20 | +10 | 0 | −10 | −20 dB |
|---|---|---|---|---|---|
| Speech | 0.986 | 0.980 | 0.967 | 0.921 | 0.739 |
| Cafeteria | 0.988 | 0.981 | 0.930 | 0.653 | 0.077 |
| Music | 0.981 | 0.946 | 0.848 | 0.629 | 0.302 |
| White | 0.989 | 0.987 | 0.975 | 0.883 | 0.352 |

Further results:

| Measure | Value |
|---|---|
| Onset precision, music noise at 0 dB / −20 dB | 0.80 / 0.37 |
| Predicted pedal presses on Bach (no pedal): clean / music −10 dB / music −20 dB | 5 / 53 / 70 |
| Recall, clean, velocity below 40 / 40 and above | 0.71 / 0.98–1.00 |
| Recall, clean, black / white keys | 0.973 / 0.978 |
| Recall, clean, 6 or more simultaneous notes | 0.89 |
| Pedal-down notes whose predicted end is closer to key release than to pedal-extended end | 81–96% per recording |
| Note ends within tolerance, pedal up / pedal down at key release | 87–100% / 39–76% per recording |
| Pedal detection, frame F1 | 0.85–0.99 per recording |

## Not included in this repository

Large files are excluded and can be regenerated:

| Excluded | Regenerated by |
|---|---|
| PianoVAM audio, MIDI, TSV | `prepare_pianovam.py` (pinned revision, checksums) |
| Raw noise sources | download LibriSpeech dev-clean, DEMAND (PCAFETER, STRAFFIC, DWASHING) and the three music recordings into `noise_sources/` |
| Noise excerpts (`noise_bank/*.wav`) | `build_noise_bank.py` (compare the new sha256 values with `noise_bank.json`) |
| Noisy mixtures | deterministic: the same recording, excerpt, seed and SNR always give the same mixture (`mix_to_wav` in `run_noise.py`); `--keep-audio` keeps them on disk |

TransKun's MIDI predictions (`predictions/`) are included, so all analyses can be rerun without transcribing again.

## Licences

- PianoVAM is licensed under CC BY-NC 4.0. The predictions and results in this repository are derived from it and are shared for non-commercial research under the same licence.
- Noise sources keep their own licences (LibriSpeech CC BY 4.0, DEMAND, public-domain music); see `noise_bank/noise_bank.json`.
- TransKun and mir_eval are used unchanged under their own licences.
