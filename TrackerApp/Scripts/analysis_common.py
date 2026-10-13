"""Shared parsing and association logic for the TrackerApp analysis scripts.

Deliberately depends only on the Python standard library: this module and the scripts that
compute numbers from it are released artefacts that a reader must be able to run without
installing anything. Plotting, which does need third-party packages, lives in plot_results.py
and consumes the CSV these scripts emit.

Two properties of the evaluation are enforced here:

* The reference trajectory is built independently of every tracker being evaluated: it is seeded
  from the persisted frame-0 ROIs and propagated from its own previous position, so it cannot
  follow any of the algorithms under test.
* Detections are assigned to markers jointly, so one detection cannot be claimed by two markers.
  Assigning each marker its own nearest detection independently would allow exactly that, and
  would hide identity swaps.

Subjects are reported by code (S1, S2, ...). The mapping from recording names to codes is read
from subject_codes.json next to this module, or from the path in the SUBJECT_CODES environment
variable; it is not part of the repository.
"""

import csv
import gzip
import json
import math
import os
import re
from collections import defaultdict

INF = float("inf")

# Camera serial numbers of the five evaluation viewpoints of the 32-camera array.
VIEWS = {"19395995": "front", "19395999": "bottom", "19467093": "right",
         "19467102": "left", "19467208": "top"}
FRONT_CAMERA = "19395995"

# Reference streams written next to the trackers; never analysed as trackers.
REFERENCE_SECTIONS = ("MultiBlobDetector", "WeightedEdgesExtractor")

# Display names of the tracker sections.
TRACKER_NAMES = {"TrackerBoosting": "Boosting", "TrackerCSRT": "CSRT", "TrackerDaSiamRPN": "DaSiamRPN",
                 "TrackerGOTURN": "GOTURN", "TrackerKCF": "KCF", "TrackerMIL": "MIL",
                 "TrackerMOSSE": "MOSSE", "TrackerMedianFlow": "MedianFlow", "TrackerNano": "Nano",
                 "TrackerTLD": "TLD", "TrackerVit": "ViT", "HybridVitCsrt": "Hybrid (ViT+CSRT)"}


# --------------------------------------------------------------------------------------------
# Subjects
# --------------------------------------------------------------------------------------------

def load_subject_codes(path=None):
    """Mapping from the subject token in a recording name to its code, e.g. {"...": "S1"}.

    Returns an empty mapping when the file is absent, which is the case for released data whose
    file names already carry the codes.
    """
    path = path or os.environ.get("SUBJECT_CODES") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "subject_codes.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def subject_of(name, codes=None):
    """Subject code for a recording or results file name.

    Accepts released names that start with the code ("S3_emotions_40_front...") and raw recording
    names ("20230223_1247_<subject>_4MarkersEmotions01", "<subject>MarkersViseme01"), whose subject
    token is translated through the mapping. Returns "unknown" if neither form matches.
    """
    base = re.sub(r"^\d{8}_\d{4}_", "", os.path.basename(name))
    coded = re.match(r"(S\d+)(?:_|$)", base)
    if coded:
        return coded.group(1)
    raw = re.match(r"([A-Za-z]+?)_?4?Markers", base)
    if not raw:
        return "unknown"
    token = raw.group(1)
    codes = load_subject_codes() if codes is None else codes
    return codes.get(token, token)


# --------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------

def parse_points(line):
    """Parses one serialised frame into a list of tuples.

    Accepts both the two-field reference format "(x;y),(x;y)," and the four-field tracker format
    "(cx;cy;w;h),...". Values are read as float: coordinates are sub-pixel, and parsing them as
    int would silently truncate them.

    Returns an empty list for the "(0;0)" placeholder that means "nothing detected".
    """
    if not line:
        return []
    points = []
    for chunk in line.split("),"):
        chunk = chunk.strip().strip("(),")
        if not chunk:
            continue
        try:
            values = [float(v) for v in chunk.split(";")]
        except ValueError:
            continue
        if len(values) >= 2:
            points.append(tuple(values))
    # "(0;0)" is the writer's placeholder for an empty detection set, not a marker at the origin.
    if len(points) == 1 and len(points[0]) == 2 and points[0] == (0.0, 0.0):
        return []
    return points


def frame_keys(section):
    """Frame keys of a per-tracker section, in order, excluding scalar entries such as "FPS"."""
    return sorted(k for k in section if k.isdigit())


EXPORT_FORMAT = "trackerapp-export-1"


def load_results(path):
    """Loads a results file written by TrackerApp or TrackerBench, or one video of the exported
    dataset (export_dataset.py), which is rebuilt into the same structure so that every script
    works on either."""
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("format") == EXPORT_FORMAT:
        return load_exported_video(path, document)
    return document


def load_exported_video(path, header):
    """Rebuilds a results document from <id>.json, <id>_trackers.csv.gz and <id>_reference.csv.gz.

    Coordinates are carried as the original text, so the rebuilt frame entries are identical to
    those of the results file the export was made from.
    """
    base = path[:-len(".json")]
    frames = int(header["frames"])
    meta = {k: v for k, v in header.items() if k not in ("format", "rois")}
    document = {"_meta": meta, "_rois": header["rois"], "_diagnostics": {}}

    detections = defaultdict(lambda: defaultdict(list))
    with gzip.open(base + "_reference.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            detections[row["stream"]][int(row["frame"])].append(f"({row['x']};{row['y']})")
    for stream in header.get("reference_streams", []):
        per_frame = detections.get(stream, {})
        document[stream] = {f"{i:04d}": (",".join(per_frame[i]) + ",") if per_frame.get(i) else "(0;0)"
                            for i in range(frames)}

    targets = defaultdict(lambda: defaultdict(dict))
    if os.path.exists(base + "_trackers.csv.gz"):          # recordings without trackers have none
        with gzip.open(base + "_trackers.csv.gz", "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                targets[row["tracker"]][int(row["frame"])][int(row["marker"])] = row
    for tracker, per_frame in targets.items():
        section, valid, switches = {}, {}, {}
        for frame in sorted(per_frame):
            rows = [per_frame[frame][m] for m in sorted(per_frame[frame])]
            key = f"{frame:04d}"
            section[key] = ",".join(f"({r['cx']};{r['cy']};{r['w']};{r['h']})" for r in rows)
            valid[key] = [r["valid"] == "1" for r in rows]
            if any(r.get("fallback") for r in rows):
                switches[key] = [r.get("fallback") == "1" for r in rows]
        document[tracker] = section
        diagnostics = {"valid": valid, "n_targets": len(header["rois"])}
        if switches:
            diagnostics["fallback_switches"] = switches
        document["_diagnostics"][tracker] = diagnostics
    return document


def roi_centres(document):
    """Frame-0 ROI centres, ordered by marker id, from the "_rois" section.

    These seed the reference trajectories, which keeps the reference independent of every tracker.
    A results file without a "_rois" section cannot be analysed this way.
    """
    rois = document.get("_rois")
    if not rois:
        return []
    ordered = sorted(rois.items(), key=lambda kv: int(kv[0]))
    # Labelled boxes carry the marker centre ("cx", "cy"); older files only the box.
    return [(r.get("cx", r["x"] + r["w"] / 2.0), r.get("cy", r["y"] + r["h"] / 2.0))
            for _, r in ordered]


def tracker_names(document):
    """Names of the tracker sections present, excluding metadata sections."""
    return [k for k in document if not k.startswith("_")]


# --------------------------------------------------------------------------------------------
# Tracking failure
# --------------------------------------------------------------------------------------------

# A tracker has lost its marker in a frame when (1) it reports failure, (2) its centre is nearer
# another marker's reference position than its own (identity swap), or (3) its centre lies more
# than LOST_BEYOND_DIAMETERS diameters of that marker from the marker's reference position. The
# tracking boxes are about three marker diameters wide, so beyond two diameters the marker is no
# longer inside the region the tracker follows: it has locked onto a skin fold or another facial
# feature. These are the failure conditions of the paper. Deviations are computed over the frames
# in which none of them holds; the share of such frames is reported with them.
LOST_BEYOND_DIAMETERS = 2.0


def marker_diameters(document):
    """Labelled marker diameters in px, ordered by marker id (a third of the box side for files
    without labelled diameters: boxes are about three marker diameters wide)."""
    ordered = sorted(document.get("_rois", {}).items(), key=lambda kv: int(kv[0]))
    return [r.get("d") or min(r["w"], r["h"]) / 3.0 for _, r in ordered]


def tracking_state(target, marker, reported_valid, reference_frame, diameter,
                   lost_beyond=LOST_BEYOND_DIAMETERS):
    """Classifies one tracker position against the reference positions of one frame.

    Returns (state, error, dx, dy, nearest_marker); state is one of
      "unscored"  the reference has no position for this marker: neither active nor lost;
      "reported"  the tracker reported failure (no deviation is computed);
      "swap"      its centre is nearer another marker's reference position than its own;
      "drift"     its centre is more than `lost_beyond` marker diameters from its own;
      "active"    it is tracking its marker.
    lost_beyond <= 0 switches the drift condition off.
    """
    reference_point = reference_frame[marker] if marker < len(reference_frame) else None
    if reference_point is None:
        return "unscored", None, None, None, None
    if not reported_valid:
        return "reported", None, None, None, None
    dx = target[0] - reference_point[0]
    dy = target[1] - reference_point[1]
    error = math.hypot(dx, dy)
    nearest, best = marker, error
    for other, point in enumerate(reference_frame):
        if other != marker and point is not None:
            distance = math.dist(target[:2], point)
            if distance < best:
                best, nearest = distance, other
    if nearest != marker:
        state = "swap"
    elif lost_beyond > 0 and error > lost_beyond * diameter:
        state = "drift"
    else:
        state = "active"
    return state, error, dx, dy, nearest


# --------------------------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------------------------

def solve_assignment(cost):
    """Minimum-cost assignment (Hungarian / Jonker-Volgenant shortest augmenting path).

    \\param cost Rectangular cost matrix, rows x columns, rows <= columns enforced by padding.
    \\returns List of length len(cost) giving the column assigned to each row, or None.

    O(n^3) and exact. At 40 markers a joint solution costs nothing and removes the
    double-assignment that per-marker nearest-neighbour matching allows.
    """
    if not cost or not cost[0]:
        return [None] * len(cost)

    rows, cols = len(cost), len(cost[0])
    size = max(rows, cols)
    # Pad to square with a large finite cost so the algorithm stays well defined.
    big = max((c for row in cost for c in row if c < INF), default=1.0) * size + 1.0
    matrix = [[(cost[r][c] if r < rows and c < cols and cost[r][c] < INF else big)
               for c in range(size)] for r in range(size)]

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)      # p[c] = row assigned to column c
    way = [0] * (size + 1)

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1

    assignment = [None] * rows
    for c in range(1, size + 1):
        r = p[c] - 1
        if r < rows and c - 1 < cols and cost[r][c - 1] < INF:
            assignment[r] = c - 1
    return assignment


def build_reference_tracks(detections_by_frame, seeds, max_jump_px=25.0):
    """Builds one reference trajectory per marker, independently of every tracker.

    \\param detections_by_frame Mapping frame key -> list of detected points.
    \\param seeds Frame-0 marker positions (the persisted ROI centres).
    \\param max_jump_px Largest plausible frame-to-frame motion; larger matches are refused.
    \\returns Mapping frame key -> list (one entry per marker) of (x, y) or None.

    Each frame is solved as a joint assignment between markers and detections, so a detection is
    claimed by at most one marker. A marker with no acceptable match is recorded as None - an
    honest gap rather than an imputed value - and its previous position is not advanced, so a
    brief occlusion does not drag the trajectory away.
    """
    previous = list(seeds)
    tracks = {}
    for frame in sorted(detections_by_frame):
        # Only detections within reach of some marker can be assigned. Dropping the others first
        # gives the same assignment and keeps it fast when a detector reports many blobs.
        points = [point for point in detections_by_frame[frame]
                  if any(math.dist(prev[:2], point[:2]) <= max_jump_px for prev in previous)]
        if not points:
            tracks[frame] = [None] * len(seeds)
            continue
        cost = [[math.dist(prev[:2], point[:2]) if math.dist(prev[:2], point[:2]) <= max_jump_px
                 else INF for point in points] for prev in previous]
        assignment = solve_assignment(cost)
        frame_result = []
        for marker_index, column in enumerate(assignment):
            if column is None:
                frame_result.append(None)
            else:
                matched = points[column][:2]
                frame_result.append(matched)
                previous[marker_index] = matched
        tracks[frame] = frame_result
    return tracks
