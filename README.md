# TrackerApp

Benchmark of the eleven OpenCV visual trackers on hand-drawn facial markers. Two programs share
one tracking core:

| Program | Purpose |
|---|---|
| `TrackerApp` | Qt desktop application: pick a video, label or review the marker boxes, run the selected trackers. |
| `TrackerBench` | Headless harness: the same trackers driven entirely by command-line arguments and box files, for scripted runs. |

`Scripts/` runs the complete benchmark over every recording and computes the comparisons from the
results; see [Scripts/README.md](TrackerApp/Scripts/README.md).

## Repository layout

```
TrackerApp.sln                     Visual Studio solution (Windows build of the GUI)
TrackerApp/
  CMakeLists.txt                   CMake build of both programs (Linux)

  Tracking core, shared by both programs and free of Qt:
  Trackers.h/.cpp                  the eleven OpenCV trackers, the ViT wrapper, the ViT+CSRT hybrid,
                                   per-marker tracking and video decoding
  MarkerBoxes.h/.cpp               marker box files, the labelling window, Blob Detection settings
                                   and the reference search region
  WeightedEdgesExtractor.h/.cpp    Point Center Weighted Extractor (PCWE)
  JsonTrackerObject.h/.cpp         results file writer (json.hpp: nlohmann/json 3.11.2)

  TrackerBench, the headless harness:
  cli_main.cpp                     command line, per-marker job scheduling, reference streams
  ThreadLocalRand.cpp              per-thread rand(), so parallel runs are reproducible

  TrackerApp, the Qt desktop application:
  main.cpp, main.qml, qml.qrc      entry point and user interface
  BackendController.h/.cpp         connection between the QML interface and the worker thread
  BackendWorker.h/.cpp             runs the trackers off the interface thread
  FrameImageProvider.h/.cpp        supplies video frames to the interface
  ImageViewer.h/.cpp               OpenGL display of OpenCV images
  TrackerApp.vcxproj*              Visual Studio project

  Scripts/                         benchmark runner, analysis, manual annotation study and dataset
                                   export (see Scripts/README.md)
```

## Build

OpenCV **4.9.0 or newer with the `opencv_contrib` modules is required** — `cv::TrackerVit` does not
exist before 4.9, and the legacy trackers live in `opencv_contrib`. The benchmark was run with
OpenCV 4.10.0. The neural trackers only run at a usable speed in Release builds.

Linux / WSL (CMake fails early if an older OpenCV is found):

```bash
cd TrackerApp
cmake -S . -B build-linux -G Ninja -D CMAKE_BUILD_TYPE=Release \
      -D OpenCV_DIR=$HOME/opt/opencv-4.10/lib/cmake/opencv4
cmake --build build-linux -j$(nproc)
```

Windows: open `TrackerApp.sln` (Visual Studio with the Qt VS Tools, Qt 5). Point the include and
library paths and the `opencv_world` library name in the project settings at an OpenCV 4.9+ build.
The project builds `TrackerApp`; `TrackerBench` is built with CMake.

## Model files

Place these in the directory you run from (`TrackerApp/`, next to the sources):

| Tracker | File | Source | SHA-256 |
|---|---|---|---|
| ViT | `vitTracker.onnx` | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack) `object_tracking_vittrack_2023sep.onnx`, renamed | `2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d` |
| GOTURN | `goturn.prototxt` | [opencv_extra](https://github.com/opencv/opencv_extra/tree/c4219d5eb3105ed8e634278fad312a1a8d2c182d/testdata/tracking) | `34e0a44494c228b8e14c80e7ea4c3a40fd8853dce86896c82c58052a8660e926` |
| GOTURN | `goturn.caffemodel` | same (assembled from the split archive) | `8de55869a37963af1ab49e4321452126cd331ae5d226dae318b8da9b90c5de53` |
| DaSiamRPN | `dasiamrpn_model.onnx` | [Dropbox](https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=0) | `e88370b85cbad914a5eb414d9d9e0820f87fd0cd89b65205a766174206c35719` |
| DaSiamRPN | `dasiamrpn_kernel_r1.onnx` | [Dropbox](https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=0) | `082c85d231b88b97a1b2a50e73b640a332c5d98d7c1d80b5da9ab534fa7a9e5b` |
| DaSiamRPN | `dasiamrpn_kernel_cls1.onnx` | [Dropbox](https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=0) | `d85b03e2aeded6cc9be945dfdc3ed6b8f4151f101e485037b6c5d5b36a6c4204` |
| Nano | `backbone.onnx` | [NanoTrack v2](https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2) | `530bdd0cd00f19afab79a863e71ba71e3312395a5dc9151af675082bdaaa2fc4` |
| Nano | `neckhead.onnx` | same | `0d8c0637be849f092cc7236cae02e55c8b9455ebe37ba50601d6115db4247cd9` |

The classical trackers (CSRT, KCF, MOSSE, MIL, TLD, Boosting, MedianFlow) need no model files.

## Marker boxes

Every tracker starts from the same initial box per marker. Boxes are labelled once per video on
frame 0 and stored in a **box file**, by convention `Labels/<recording folder>/<video stem>.json`:

```json
{
  "_meta": { "video": "...", "box_scale": 3.0, "n_boxes": 40 },
  "_rois": { "0": { "x": 731, "y": 588, "w": 43, "h": 43, "cx": 752.4, "cy": 609.8, "d": 14.2, "adjusted": false } }
}
```

`x, y, w, h` is the box; `cx, cy` and `d` are the marker centre and diameter it was built from. A
box is a square of side 3 × the marker diameter, centred on the marker, unless the operator
reshaped it (`"adjusted": true`), for example around a marker that is not round. `--roi-scale`
rebuilds the squares at another multiple of the diameter and scales reshaped boxes about their
centre in the same proportion.

Boxes are labelled in an OpenCV window opened by both programs, pre-filled with the markers Blob
Detection finds on frame 0; press `h` in the window for its keys.

## Reproducing the published results

The data of the article are attached to the
[v1.0-sensors release](https://github.com/PhotonicTree/TrackerApp/releases/tag/v1.0-sensors) as
`TrackerApp-data.zip`: the initial boxes, every reference detection and every tracker position in
every frame, and the manual annotations (layout described in its README). The analysis scripts
recompute the results from these files alone, with Python 3 and its standard library; neither the
videos nor OpenCV are needed. Table and section numbers refer to the article.

```bash
unzip TrackerApp-data.zip                      # creates TrackerApp-data/
cd TrackerApp/Scripts
D=../../TrackerApp-data                        # where the archive was unpacked

python3 analyse_tracking.py --results $D/sparse/*.json            # Table 4: pooled table at the end
python3 aggregate_subjects.py --glob "$D/sparse/*_front.json" --out-prefix front
python3 subject_statistics.py front_trajectories.csv              # Section 5.2: intervals and paired tests
python3 analyse_tracking.py --results $D/sweep/s1.5/*.json        # Table 5, once per scale s1.5 ... s3.0
python3 compare_hybrid.py "$D/dense/*.json"                       # final retention, mis-assignment (Table 7)
python3 analyse_tracking.py --results $D/dense/*.json             # Table 7: deviations, pooled table
python3 validate_reference.py --results $D/glued/*.json           # Section 5.1; also sparse/ and dense/
python3 analyse_annotations.py --points $D/annotation_points.csv --data $D   # Section 5.1: manual annotation
```

Deviations are in pixels; the article converts them at 0.24 mm/pixel. `analyse_tracking.py` also
writes the per-frame states and deviations (`per_frame_errors.csv`, several hundred MB for the
40-marker files). Two quantities need material that is not released: the annotators' repeatability
(it needs the image patches) and the execution latency of Table 6 (measured separately on one
thread). Re-running the trackers themselves needs the recordings, which are not public, OpenCV
4.10.0 with `opencv_contrib` and the model files above.

## Two ways to run

### 1. Interactively, in the GUI

1. Start `TrackerApp` from the directory holding the model files.
2. Pick a video with `...` and switch on the trackers to run.
3. **Run Tracking For Selected Trackers**: the labelling window opens (pre-filled with the saved
   boxes if the video has a box file); press Enter, and the boxes are saved and every selected
   tracker runs. Results go to `Results/<recording folder>/<video stem>.json`.

**Label Boxes Only** labels and saves the boxes without running anything, which prepares videos
for scripted runs. With **Reuse saved boxes without review** on, a video that already has a box
file runs straight from it.

### 2. As a script, from boxes labelled beforehand

```bash
cd TrackerApp
# label one video (or use the GUI); saves Labels/<folder>/<video stem>.json
./build-linux/TrackerBench --label --video Recordings/<folder>/Preview_19395995.mp4 --expect 40

# run trackers from that box file
./build-linux/TrackerBench --video Recordings/<folder>/Preview_19395995.mp4 --label-dir Labels \
    --trackers CSRT,KCF,ViT --out Results/run.json
```

For the whole benchmark — every recording, every view, every tracker, resumable — use
`Scripts/run_benchmark.py` (label → run → analyse), described in
[Scripts/README.md](TrackerApp/Scripts/README.md).

## TrackerBench

```
TrackerBench --label --video <path> [--label-dir <dir> | --save-rois <file>] [--expect n]
TrackerBench --video <path> [--rois <file> | --label-dir <dir> | --roi-scale <s>] [options]
```

| Option | Meaning |
|---|---|
| `--video <path>` | Sequence to track (required). |
| `--label` | Labelling mode: label frame 0 and save the box file. Exit code 0 saved, 4 skipped, 5 stop requested. |
| `--save-rois <file>` | Box file written by `--label` (default: by convention under `--label-dir`, default `Labels`). |
| `--rois <file>` | Box file, or a results file whose `_rois` section is reused. |
| `--label-dir <dir>` | Find the box file by convention under `<dir>`. |
| `--roi-scale <s>` | With a box file: rebuild every box at side s × its marker diameter (region-size sweep). Without: boxes around the frame-0 Blob Detection. |
| `--trackers a,b,c` | Trackers to run (default: all eleven and `Hybrid`). `--list` prints the names, including the reference streams `BlobDetector`, `BlobDetectorGlued`, `PCWE` and `PCWEGlued`. |
| `--targets a:b` | Track only boxes a … b−1 (all boxes are still recorded); `run_benchmark.py` tracks one marker per process this way. |
| `--expect <n>` | Fail unless exactly n boxes are available. |
| `--max-frames <n>` | Truncate the sequence for quick checks. |
| `--reference-fixed` | Reference streams with the fixed presets over the whole frame instead of the labelled-box settings (see below). |
| `--threads <n>` | OpenCV thread count (default 1, the single-thread latency protocol; -1 for the OpenCV default). |
| `--jobs <n>` | Per-marker jobs on n threads instead of the frame-by-frame run: every (tracker, marker) pair is tracked over the whole sequence as one job, reference streams in frame ranges. Each job starts from the default random state, so results are identical for any n and equal to tracking each marker alone. OpenCV is single-threaded inside the jobs; no latency is measured. |
| `--limit a=n,...` | With `--jobs`: at most n concurrent jobs of tracker a. Defaults `GOTURN=3,DaSiamRPN=6` (large networks per marker) and `MIL=1` (OpenCV's MIL shares scratch buffers between instances). |
| `--show` | Display tracking windows (off by default). |
| `--out <file>` | Output JSON path; the directory is created if needed. |

### Reference streams

`BlobDetector` (Blob Detection) and `PCWE` detect every marker in every frame, independently of the
trackers. When the boxes come from a box file, their detector settings are derived from the labelled
markers of that video and detection is restricted to the face region:

- area window from 0.4 × the area of the 10th-percentile labelled diameter to 2.5 × the area of the
  90th-percentile diameter, so it follows the apparent marker size of each camera (markers are about
  6 px across from the bottom camera and 13–15 px from the front);
- circularity ≥ 0.5, convexity ≥ 0.75, inertia ratio ≥ 0.1; thresholds 20–150 in steps of 1;
- search region: the bounding rectangle of the labelled boxes plus 200 px on every side.

The settings and region used are written to `_meta.reference_detector`. `--reference-fixed` uses the
fixed drawn- and glued-marker presets over the whole frame instead. The analysis scripts associate
the detections with markers starting from the labelled centres, each marker following its own
previous reference position, so the reference never depends on a tracker.

### Results file layout

Each tracker has one section keyed by its name (`TrackerCSRT`, …) holding one entry per frame
(`"0001"`, `"0002"`, …; frame 0 initialises the trackers) and an `FPS` entry. The reference
streams write `MultiBlobDetector` (Blob Detection) and `WeightedEdgesExtractor` (PCWE). Three
sections document the run:

- `_meta` — OpenCV version, thread count, CPU features, video, frame count, box source and the
  detector presets of the reference streams;
- `_rois` — the initial boxes (as in the box file), so a run is reproducible;
- `_diagnostics.<Tracker>` — `frame_times_us` (per-frame timings, from which the latency means and
  standard deviations are computed), `n_targets`, `valid` (the per-frame success flags the OpenCV
  APIs return) and `fps`.

**Coordinates are sub-pixel.** A tracker frame entry is `(cx;cy;w;h),(cx;cy;w;h),…` in box order
and a reference entry is `(x;y),(x;y),…`, with three decimal places; parsers must use
`float(...)`, not `int(...)`.

## Hybrid pipeline

`HybridTracker` implements a cooperative scheme: ViT as the primary state-estimator with CSRT as
the fallback state-corrector. Both advance every frame; an anomaly in the primary — a failed
update, a box whose area has drifted outside `[0.5, 2.0]` of its initial size, or a centre jump
beyond 40 px — hands that frame to the fallback and re-initialises ViT from the fallback's box.
Every switch is recorded under `_diagnostics.HybridVitCsrt.fallback_switches`.

## License

MIT — see [LICENSE](LICENSE). `json.hpp` is nlohmann/json, MIT.
