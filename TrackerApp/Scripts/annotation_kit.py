"""Builds the manual-annotation kit that validates the reference independently of any detector.

  python annotation_kit.py --results ../Results/benchmark --recordings <rec> --out ../Results/annotation

Writes into --out:
  annotate.html   the annotation page: one self-contained file with the marker patches embedded.
                  Open it in a browser, offline; it shows nothing but the patches, saves progress
                  in the browser and exports the clicks as a JSON file. Give the same file to
                  every annotator.
  manifest.json   which video, frame and marker every patch comes from, and where the patch lies
                  in the frame. It names the recordings and is not distributed;
                  analyse_annotations.py reads it.

Sample (all parameters are options):
  - every marker of every 4-marker video (14 sequences x 5 views) in two frames: one near the
    start, where the face is close to neutral, and the frame of largest marker displacement;
  - 10 random markers of every 40-marker video in the same two frames;
  - 100 marker-frames where the two reference streams disagree by more than 2 px or only one of
    them found the marker, drawn at random from all of these videos;
  - 10 % of all patches again, cut with a different offset, to measure each annotator's
    repeatability.

Each patch is about four marker diameters wide (48-96 px) and shown enlarged. The patch is cut
around the approximate marker position with a random offset of up to a fifth of its size, so the
marker is not at its centre and the annotator cannot take the detector's position from it; the
annotator sees no detector output. Patch ids are random and carry no information.

Needs OpenCV for Python (pip install opencv-python); otherwise standard library.
"""

import argparse
import base64
import datetime
import glob
import json
import math
import os
import random
import re
import string
import sys

import cv2

from analysis_common import (FRONT_CAMERA, VIEWS, build_reference_tracks, frame_keys, load_results,
                             parse_points, roi_centres, subject_of, load_subject_codes)

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "annotate_template.html")


def describe_folder(folder):
    sequence = "visemes" if "Viseme" in folder else "emotions"
    layout = "4" if "4Marker" in folder else "40"
    return sequence, layout


def reference_tracks(document, section):
    detections = {k: parse_points(document[section][k]) for k in frame_keys(document[section])}
    return build_reference_tracks(detections, roi_centres(document))


_tracks = {}


def video_tracks(path):
    """(document, labelled centres, Blob reference, PCWE reference) of one results file, built once."""
    if path not in _tracks:
        document = load_results(path)
        _tracks[path] = (document, roi_centres(document), reference_tracks(document, "MultiBlobDetector"),
                         reference_tracks(document, "WeightedEdgesExtractor"))
    return _tracks[path]


def approximate_position(blob, pcwe, seeds, frame, marker):
    """Where to cut the patch: the Blob reference, else PCWE, else the nearest Blob position in
    time, else the labelled position."""
    for track in (blob, pcwe):
        point = track.get(f"{frame:04d}", [None] * len(seeds))[marker]
        if point:
            return point, "blob" if track is blob else "pcwe"
    for distance in range(1, 60):
        for f in (frame - distance, frame + distance):
            point = blob.get(f"{f:04d}", [None] * len(seeds))[marker] if f >= 0 else None
            if point:
                return point, "nearby"
    return seeds[marker], "labelled"


def peak_frame(blob, seeds):
    """Frame of largest mean displacement of the referenced markers from their start positions."""
    start = blob.get("0000", seeds)
    best, best_value = 0, -1.0
    for key, points in blob.items():
        moved = [math.dist(p, q) for p, q in zip(points, start) if p and q]
        if len(moved) >= max(1, len(seeds) // 2):
            value = sum(moved) / len(moved)
            if value > best_value:
                best, best_value = int(key), value
    return best


def new_id(rng, taken):
    while True:
        candidate = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        if candidate not in taken:
            taken.add(candidate)
            return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="run_benchmark.py output directory")
    parser.add_argument("--recordings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dense-markers", type=int, default=10, help="markers per 40-marker video and frame")
    parser.add_argument("--disagreements", type=int, default=100)
    parser.add_argument("--disagreement-px", type=float, default=2.0)
    parser.add_argument("--repeat", type=float, default=0.10, help="share of patches shown twice")
    parser.add_argument("--only", help="regular expression on the results file names (tests)")
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    codes = load_subject_codes()
    files = sorted(glob.glob(os.path.join(args.results, "sparse", "*.json")) +
                   glob.glob(os.path.join(args.results, "dense", "*.json")))
    if args.only:
        files = [f for f in files if re.search(args.only, os.path.basename(f))]
    if not files:
        raise SystemExit("No results files found.")

    points, candidates = [], []
    for path in files:
        document, seeds, blob, pcwe = video_tracks(path)
        name = os.path.basename(path)[:-len(".json")]
        folder, _, camera = name.partition("__")
        sequence, layout = describe_folder(folder)
        video_id = f"{subject_of(folder, codes)}_{sequence}_{layout}_{VIEWS[camera]}"
        diameters = [document["_rois"][k]["d"] for k in sorted(document["_rois"], key=int)]
        frames = len(blob)
        video = os.path.join(args.recordings, folder, f"Preview_{camera}.mp4")
        common = dict(results=os.path.relpath(path, args.results), video=video, video_id=video_id,
                      layout=layout, view=VIEWS[camera])

        near_start = rng.randint(5, max(5, frames // 10))
        chosen = {"start": near_start, "peak": peak_frame(blob, seeds)}
        markers = list(range(len(seeds)))
        for stratum, frame in chosen.items():
            sample = markers if layout == "4" else sorted(rng.sample(markers, min(args.dense_markers, len(markers))))
            for marker in sample:
                points.append(dict(common, frame=frame, marker=marker, stratum=stratum, diameter=diameters[marker]))

        for key in blob:
            for marker in markers:
                a, b = blob[key][marker], pcwe.get(key, [None] * len(seeds))[marker]
                if (a is None) != (b is None) or (a and b and math.dist(a, b) > args.disagreement_px):
                    candidates.append(dict(common, frame=int(key), marker=marker, stratum="disagreement",
                                           diameter=diameters[marker]))
        print(f"{video_id:<26} {frames:4d} frames, {len(seeds):2d} markers, start {near_start}, peak {chosen['peak']}")

    # Disagreements, half from each layout: the 40-marker videos have far more marker-frames.
    taken_keys = {(p["video"], p["frame"], p["marker"]) for p in points}
    candidates = [c for c in candidates if (c["video"], c["frame"], c["marker"]) not in taken_keys]
    pools = {layout: [c for c in candidates if c["layout"] == layout] for layout in ("4", "40")}
    take4 = min(len(pools["4"]), max(args.disagreements // 2, args.disagreements - len(pools["40"])))
    points += rng.sample(pools["4"], take4)
    points += rng.sample(pools["40"], min(len(pools["40"]), args.disagreements - take4))
    repeats = [dict(p, stratum=p["stratum"], repeat=True) for p in rng.sample(points, round(args.repeat * len(points)))]
    points += repeats

    # Cut the patches, reading every video once from the start.
    ids, by_video = set(), {}
    for point in points:
        by_video.setdefault(point["video"], []).append(point)
    items, manifest = [], []
    for video, wanted in by_video.items():
        _, seeds, blob, pcwe = video_tracks(os.path.join(args.results, wanted[0]["results"]))
        needed = {p["frame"] for p in wanted}
        capture = cv2.VideoCapture(video)
        index, frames = 0, {}
        while needed - frames.keys():
            ok, image = capture.read()
            if not ok:
                break
            if index in needed:
                frames[index] = image
            index += 1
        for point in wanted:
            image = frames.get(point["frame"])
            if image is None:
                raise SystemExit(f"Frame {point['frame']} not readable in {video}")
            centre, source = approximate_position(blob, pcwe, seeds, point["frame"], point["marker"])
            size = int(min(96, max(48, round(4 * point["diameter"]))))
            dx, dy = (rng.uniform(-0.2, 0.2) * size for _ in range(2))
            x0 = int(round(centre[0] + dx - size / 2))
            y0 = int(round(centre[1] + dy - size / 2))
            patch = cv2.copyMakeBorder(image, size, size, size, size, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            patch = patch[y0 + size:y0 + 2 * size, x0 + size:x0 + 2 * size]
            ok, png = cv2.imencode(".png", patch)
            patch_id = new_id(rng, ids)
            items.append({"id": patch_id, "size": size, "png": base64.b64encode(png.tobytes()).decode("ascii")})
            manifest.append(dict({k: v for k, v in point.items() if k != "diameter"}, id=patch_id, x0=x0, y0=y0,
                                 size=size, centre_source=source, repeat=point.get("repeat", False)))

    # The repeats refer to the patch of the same marker-frame that is not a repeat.
    first = {}
    for entry in manifest:
        if not entry["repeat"]:
            first[(entry["video"], entry["frame"], entry["marker"])] = entry["id"]
    for entry in manifest:
        if entry["repeat"]:
            entry["repeat_of"] = first[(entry["video"], entry["frame"], entry["marker"])]

    os.makedirs(args.out, exist_ok=True)
    kit_id = "kit-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rng.shuffle(items)
    with open(TEMPLATE, encoding="utf-8") as handle:
        page = handle.read()
    page = page.replace("/*KIT_DATA*/null", json.dumps({"kit": kit_id, "items": items}, separators=(",", ":")))
    with open(os.path.join(args.out, "annotate.html"), "w", encoding="utf-8") as handle:
        handle.write(page)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"kit": kit_id, "seed": args.seed, "patches": manifest}, handle, indent=1)

    strata = {}
    for entry in manifest:
        key = ("repeat" if entry["repeat"] else entry["stratum"]) + " / " + entry["layout"] + " markers"
        strata[key] = strata.get(key, 0) + 1
    print(f"\n{len(manifest)} patches ({kit_id}):")
    for key, count in sorted(strata.items()):
        print(f"  {key:<32} {count}")
    size_mb = os.path.getsize(os.path.join(args.out, "annotate.html")) / 1e6
    print(f"wrote {os.path.join(args.out, 'annotate.html')} ({size_mb:.1f} MB) and manifest.json")


if __name__ == "__main__":
    main()
