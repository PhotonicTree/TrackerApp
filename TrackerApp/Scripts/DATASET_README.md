# Tracker outputs and reference trajectories for hand-drawn facial markers

Outputs of the eleven OpenCV visual trackers (and a ViT + CSRT hybrid) tracking hand-drawn markers
on deforming faces, with the two reference streams they are evaluated against. Seven subjects
(S1–S7) performed two movement sets, "Emotions" and "Visemes", filmed by a synchronised 32-camera
array at 60 frames per second; five of the cameras are used here.

**No images are included.** The participants did not consent to the public release of
identifiable video. The files hold coordinates only, and subjects are identified by code.

The code that produced these files and computes every result from them is in the TrackerApp
repository (`Scripts/`).

## Contents

| Path | Content |
|---|---|
| `videos.csv` | One row per exported video: id, condition, box scale, subject, sequence, layout, view, frame count, number of markers, number of trackers, file. |
| `sparse/` | 4 markers per face, five views (front, left, right, top, bottom). |
| `dense/` | 40 markers per face, frontal view. |
| `glued/` | 3 mm glued semi-spherical markers, five views; reference streams only, no tracker outputs. |
| `sweep/s<scale>/` | 4 markers, frontal view, with the initial boxes rebuilt at 1.5, 2.0, 2.5 and 3.0 × the marker diameter. |
| `annotation_points.csv` | Manual marker centres from two annotators, with the Blob Detection reference position of the same marker-frames (see below). |

Each video is three files, named by its id `<subject>_<sequence>_<layout>_<view>` (two for the glued
videos, which have no tracker outputs):

- **`<id>.json`** — metadata (subject, sequence, layout, view, condition, frame count, OpenCV
  version, the reference detector settings and search region, list of trackers) and `rois`, the initial box of every
  marker: `x, y, w, h` (box) and `cx, cy, d` (marker centre and diameter the box was built from).
  Boxes are squares of side `box_scale` × `d` centred on the marker unless `adjusted` is true, in
  which case the operator reshaped the box. Marker numbers are the keys of `rois`.
- **`<id>_trackers.csv.gz`** — one row per tracker, frame and marker:

  | Column | Meaning |
  |---|---|
  | `tracker` | `TrackerBoosting`, `TrackerCSRT`, `TrackerDaSiamRPN`, `TrackerGOTURN`, `TrackerKCF`, `TrackerMIL`, `TrackerMOSSE`, `TrackerMedianFlow`, `TrackerNano`, `TrackerTLD`, `TrackerVit`, `HybridVitCsrt` |
  | `frame` | Frame index; frame 0 initialises the trackers, so tracker rows start at 1. |
  | `marker` | Marker number, as in `rois`. |
  | `cx`, `cy`, `w`, `h` | Centre and size of the tracker's box. |
  | `valid` | 1 if the tracker reported success for this frame, 0 otherwise. |
  | `fallback` | Hybrid only: 1 if CSRT corrected ViT in this frame. Empty for other trackers. |

- **`<id>_reference.csv.gz`** — one row per detection: `stream`, `frame`, `x`, `y`. The stream
  `MultiBlobDetector` is the Blob Detection reference against which deviations are measured;
  `WeightedEdgesExtractor` is the Point Center Weighted Extractor (PCWE) used to check it; it is
  included for the videos on which the two are compared, the glued markers and
  `S6_emotions_4_front` (drawn markers).
  Detections are not assigned to markers; frames without detections have no rows.

Coordinates are pixels in the 1080 × 1920 (portrait) frame, origin at the top-left corner, x to
the right and y down, with three decimal places.

**`annotation_points.csv`** — independent validation of the reference. Two annotators marked the
marker centre in enlarged image patches shown without any detector output. One row per annotated
marker-frame:

| Column | Meaning |
|---|---|
| `video_id`, `frame`, `marker` | The marker-frame, as in the other files. |
| `stratum` | `start` (a frame near the start of the sequence), `peak` (the frame of the marker's largest displacement) or `disagreement` (a marker-frame in which Blob Detection and PCWE differ by more than 2 px or only one located the marker; not representative). |
| `layout`, `view` | 4- or 40-marker layout, camera view. |
| `annotator_1_x`, `annotator_1_y`, `annotator_2_x`, `annotator_2_y` | The two annotators' centres; empty when the annotator marked the marker as not visible. |
| `blob_x`, `blob_y` | The Blob Detection reference position of that marker in that frame; empty when the reference has none. |

Patches repeated to measure each annotator's repeatability are not included; the repeatability is
reported in the article.

## Reproducing the results

Deviations, failure and identity-swap labels, retention, failure timing and all statistics are
derived from these files by the scripts and are not stored. Every script accepts the `<id>.json`
files directly, for example:

```bash
python3 aggregate_subjects.py --glob "data/sparse/*.json"   # deviation from the reference
python3 compare_hybrid.py "data/dense/*.json"               # retention, 40 markers
python3 validate_reference.py --results data/glued/*.json   # Blob Detection versus PCWE
```

The trackers ran with OpenCV 4.10.0 and its `opencv_contrib` modules, at their default
parameters; the neural trackers used the model files listed, with their hashes, in the TrackerApp
README. Per-frame timings are not included: the markers were tracked in parallel processes, so
their timings are not latencies. Latency was measured separately, on a single thread.

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0); see `LICENSE`.
