# Scripts

Everything needed to rerun the full benchmark and compute the comparisons from its results.
Standard library Python 3 only; nothing to install.

| Script | Purpose |
|---|---|
| `run_benchmark.py` | Plans, labels and runs the complete benchmark with `TrackerBench`; resumable. |
| `export_dataset.py` | Exports the results as the publishable dataset (coded subjects, no images, no paths). |
| `analyse_tracking.py` | Per-frame deviations from the reference and per-tracker summary metrics. |
| `aggregate_subjects.py` | Per-trajectory and per-subject aggregation, bootstrap intervals over subjects. |
| `compare_hybrid.py` | Retention, retention area, mis-assignment and deviation per tracker, averaged per subject; the hybrid against its components. |
| `validate_reference.py` | Agreement between the Blob Detection and PCWE reference streams, all markers, every frame. |
| `show_reference.py` | Visual check of the reference: the reference alone on white beside the same points on the recording (needs `opencv-python`). |
| `annotation_kit.py` | Builds the manual-annotation kit: a stratified sample of marker-frames, cut as anonymous patches into one offline page (`annotate_template.html`), plus a private manifest (needs `opencv-python`). |
| `analyse_annotations.py` | Manual centres against Blob Detection, PCWE and the trackers; inter-annotator agreement and repeatability. |
| `analysis_common.py` | Shared parsing, tracker-independent reference construction, exact assignment, subject codes. |

## Running the full benchmark

Build `TrackerBench` and place the model files first (see the top-level README). Then, with
`<rec>` the directory that holds one folder per recording (`Preview_<camera>.mp4` files inside):

```bash
cd TrackerApp/Scripts
python3 run_benchmark.py plan  --recordings <rec>     # jobs, and which videos still need boxes
python3 run_benchmark.py label --recordings <rec>     # label them one after another
python3 run_benchmark.py run   --recordings <rec>     # run everything (resumable)
python3 run_benchmark.py status --recordings <rec>    # progress, also while running
```

Boxes can equally be labelled in the GUI (**Label Boxes Only**); both write the same box files under
`TrackerApp/Labels/`. A video without a box file is skipped by `run` and listed by `plan`.

The recordings are classified by folder name:

| Condition | Recordings | Views | Streams |
|---|---|---|---|
| `sparse` | `…4Markers…` | front, left, right, top, bottom | 11 trackers, Hybrid, Blob Detection, PCWE |
| `dense` | `…Markers…` (40 markers) | front | 11 trackers, Hybrid, Blob Detection, PCWE |
| `glued` | `…Dot…` (glued markers) | five views | Blob Detection and PCWE, glued preset |
| `sweep` | `…4Markers…` | front | 11 trackers and Blob Detection, boxes at 1.5, 2.0, 2.5 and 3.0 × marker diameter (reshaped boxes scaled in proportion) |

Results are written to `TrackerApp/Results/benchmark/<condition>/<recording folder>__<camera>.json`
(`sweep/s<scale>/…` for the sweep), with a timestamped `PROGRESS.log` alongside.

**Time and memory.** Every marker is tracked by its own tracker instance, so `run` starts one
`TrackerBench` process per (video, tracker, marker), plus one per video for the reference streams,
and runs up to `--processes` of them at a time (default 12 of 20 logical CPUs; on the i9-13900H
under WSL the throughput is the same from 8 to 16 processes and half as high with all 20). A
tracker process decodes the video itself and needs 0.15–0.4 GB, GOTURN about 1 GB; the reference
process holds the decoded video, about 4.4 GB for 700 frames. Processes start only while their
measured peak memory fits in what was available at the start minus `--memory-reserve` (1.5 GB).
Every process starts from the default random state and shares nothing with the others, so each
result is what tracking that marker alone gives; this was checked for every tracker against
runs in one process, with 1 and 20 threads. The 40-marker pilot video took about 50 minutes and
the complete benchmark about half a day. No latency is measured, since the processes compete for
the CPU; `run --frame-by-frame` runs the trackers frame by frame, one process at a time, as the
single-thread latency measurement (`TrackerBench --threads 1`) does. To keep a run going after the
terminal is closed:

```bash
setsid nohup python3 run_benchmark.py run --recordings <rec> > run.out 2>&1 &
```

Stopping the runner (`pkill -f "run_benchmark.py run"`) also stops its processes. A crash, a stop
or an out-of-memory kill loses at most the markers that were running; running `run` again starts
only what is missing. `run --references-only` recomputes just the reference streams (Blob
Detection, PCWE) of every video and merges them with the tracker outputs already written, without
running the trackers again. `--only <regex>` restricts any command to matching videos, `--conditions` to a subset of
conditions, and `--max-frames <n>` truncates every video for a quick end-to-end check.

## Computing the comparisons

```bash
R=../Results/benchmark
python3 aggregate_subjects.py --glob "$R/sparse/*.json" --out-prefix sparse   # deviation, 4 markers, 5 views
python3 aggregate_subjects.py --glob "$R/dense/*.json"  --out-prefix dense    # deviation, 40 markers, frontal
python3 compare_hybrid.py "$R/dense/*.json"                                   # retention, 40 markers
python3 compare_hybrid.py "$R/sparse/*.json"                                  # retention, 4 markers
python3 validate_reference.py --results $R/glued/*.json  --out-csv glued_agreement.csv
python3 validate_reference.py --results $R/sparse/*.json --out-csv drawn_agreement.csv
for s in 1.5 2.0 2.5 3.0; do
  python3 aggregate_subjects.py --glob "$R/sweep/s$s/*.json" --out-prefix sweep_s$s
done
python3 analyse_tracking.py --results $R/dense/*.json                         # per-frame CSV
```

Both conditions get deviation and failure measures from the same runs, so deviation and retention
can be compared within a condition.

A tracker has lost its marker in a frame when it reports failure, when its centre is nearer another
marker's reference position than its own, or when it lies more than two marker diameters from its
own marker (the boxes are about three diameters wide, so the marker is then outside the tracked
region). Deviations are computed over the remaining (active) frames and reported with the share of
such frames; retention counts a marker as lost from the first failure on. `--lost-beyond` changes
the two-diameter threshold in all three scripts (0 switches that condition off).

## Checking the reference visually

```bash
python3 show_reference.py --results $R/sparse/<video>.json --show                 # interactive
python3 show_reference.py --results $R/sparse/<video>.json --trackers CSRT --pcwe  # writes <video>_reference.mp4
python3 show_reference.py --results $R/sparse/<video>.json --snapshot 150          # one frame as PNG
```

Left: the reference alone on white (coloured dot = marker n, grey = detection not assigned to any
marker, red `n?` = no reference for marker n in that frame). Right: the same on the recording. The
reference is built by the same code as in the analysis. The recording path is read from the results
file; pass `--video` when it has moved or when reading the exported dataset, which stores no paths.
The output shows faces, so keep it out of the repository (`Results/` is ignored).

## Validating the reference by manual annotation

```bash
python annotation_kit.py --results ../Results/benchmark --recordings <rec> --out ../Results/annotation
# every annotator opens ../Results/annotation/annotate.html, annotates, presses Export
python3 analyse_annotations.py --kit ../Results/annotation --results ../Results/benchmark annotations_*.json
```

The kit samples every marker of the 4-marker videos and 10 random markers of each 40-marker video
in two frames (near the start, and at the largest displacement), adds marker-frames where Blob
Detection and PCWE disagree, and repeats 10 % of them. Each marker-frame is shown as a small patch
cut around it with a random offset, without any detector output; the annotator clicks the marker
centre or marks it as not visible. The page works offline, keeps progress in the browser and
exports the clicks. `manifest.json` names the recordings and stays private; the analysis writes
`annotation_points.csv` with coded video ids and annotators, which can be released.

## Exporting the dataset

```bash
python3 export_dataset.py --results ../Results/benchmark --out ../../data
```

writes the publishable dataset to `data/` at the repository root. Each video becomes `<id>.json`
(metadata and initial boxes), `<id>_trackers.csv.gz` (one row per tracker, frame and marker) and
`<id>_reference.csv.gz` (Blob Detection and PCWE detections), with coded ids such as
`S3_visemes_40_front`, plus `videos.csv`, the manual annotations (`annotation_points.csv`, if
`analyse_annotations.py` has been run), a README and a CC BY 4.0 licence (`DATASET_README.md`,
`DATASET_LICENSE`). Every analysis script reads `<id>.json` directly, and gives the same numbers
on the dataset as on the raw results. Deviations, failure labels, retention and statistics are
derived by the scripts and are not stored. The export stops if a subject has no code, and it
fails if a subject token, folder name or local path would reach a data file. The complete
benchmark exports to about 50 MB.

## Subject codes

Outputs identify subjects by code (S1, S2, …), never by name. The recording folder names contain
a subject token, which is translated through a private mapping file, `subject_codes.json` in this
directory (or the path in the `SUBJECT_CODES` environment variable), for example
`{"<token>": "S1"}`. The file is listed in `.gitignore` and must not be committed. Files whose names
already start with a code (`S3_…`) need no mapping.

## Design decisions

**The reference does not depend on any tracker.** Reference trajectories are seeded from the
marker centres in the box file and propagated from their own previous position, never from a
tracker's output, so the reference cannot follow one of the algorithms under test.

**Detections are assigned jointly.** Giving each marker its own nearest detection would let one
detection serve two markers and hide identity swaps. Assignment is an exact minimum-cost matching
(Hungarian / Jonker–Volgenant), verified against brute force.

**Lost frames are excluded, never imputed.** Two deviations are reported side by side:

- `mean_error_px` — conditional on active tracking, with an `exposure` column saying what
  fraction of frames it covers;
- `penalised_error_px` — over **all** frames, charging a lost frame the marker's ROI diagonal
  (override with `--penalty-px`).

They can rank trackers very differently: a tracker that loses its target early can show a small
conditional deviation computed over few frames, and a large penalised one.

**Identity swaps are detected automatically.** Each tracker target is scored against its own
fixed reference marker; a frame where the target sits closer to a different marker counts as
mis-assigned, and in `compare_hybrid.py` ends that marker's retention.

**Markers the reference never sees are not scored.** Retention and its area are computed over
the frames in which the reference detects the marker; a marker it never detects is left out
rather than counted as retained or lost.

**Coordinates are parsed as float.** The C++ side emits sub-pixel values.

**The reference comparison keeps every frame.** Frames in which Blob Detection and PCWE disagree
by more than the gate are marked, not dropped, and the agreement statistics are printed both with
and without them.
