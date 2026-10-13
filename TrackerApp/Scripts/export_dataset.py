"""Exports the benchmark results as the publishable dataset: coded subjects, no images, no paths.

  python3 export_dataset.py --results ../Results/benchmark --out ../../data

For every results file written by run_benchmark.py this writes three files, named by a coded
video id such as S3_visemes_40_front:

  <id>.json               metadata and the initial boxes
  <id>_trackers.csv.gz    one row per tracker, frame and marker: tracker, frame, marker, cx, cy,
                          w, h, valid, fallback
  <id>_reference.csv.gz   one row per detection: stream (MultiBlobDetector = Blob Detection,
                          WeightedEdgesExtractor = PCWE), frame, x, y

plus videos.csv (one row per exported video), annotation_points.csv (the manual annotations, when
analyse_annotations.py has been run), README.md and LICENSE. Every analysis script reads
<id>.json directly (analysis_common.load_results), so the published numbers can be recomputed
from the dataset alone. Deviations, failure and identity-swap labels, retention and statistics
are derived by the scripts and are therefore not stored.

Subjects are written as codes only. Recording folders carry a subject token, mapped through the
private subject_codes.json; the export stops if a token has no code, and it checks that no folder
name, subject token or local path reaches an exported file.

Standard library only.
"""

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import statistics
import sys

from analysis_common import (EXPORT_FORMAT, REFERENCE_SECTIONS, VIEWS, frame_keys, load_subject_codes,
                             subject_of)

CHUNK = re.compile(r"\(([^)]*)\)")


def describe_folder(folder):
    """Sequence and layout of a recording folder, from its name."""
    sequence = "visemes" if "Viseme" in folder else "emotions"
    if "Dot" in folder:
        layout = "glued"
    elif "4Marker" in folder:
        layout = "4"
    else:
        layout = "40"
    return sequence, layout


def results_files(results_dir):
    """(condition, scale or None, path) for every results file under the run_benchmark layout."""
    found = []
    for condition in ("sparse", "dense", "glued"):
        directory = os.path.join(results_dir, condition)
        if os.path.isdir(directory):
            found += [(condition, None, os.path.join(directory, f))
                      for f in sorted(os.listdir(directory)) if f.endswith(".json")]
    sweep = os.path.join(results_dir, "sweep")
    if os.path.isdir(sweep):
        for scale_dir in sorted(os.listdir(sweep)):
            match = re.fullmatch(r"s(\d+(?:\.\d+)?)", scale_dir)
            if not match:
                continue
            directory = os.path.join(sweep, scale_dir)
            found += [("sweep", float(match.group(1)), os.path.join(directory, f))
                      for f in sorted(os.listdir(directory)) if f.endswith(".json")]
    return found


def chunks(line):
    """Entries of one serialised frame as lists of coordinate strings, without the empty placeholder."""
    return [c.split(";") for c in CHUNK.findall(line or "") if c != "0;0"]


def export_one(condition, scale, path, out_dir, codes):
    name = os.path.basename(path)[:-len(".json")]
    folder, _, camera = name.partition("__")
    subject = subject_of(folder, codes)
    if not re.fullmatch(r"S\d+", subject):
        raise SystemExit(f"No subject code for {folder!r}: add its subject token to subject_codes.json.")
    if camera not in VIEWS:
        raise SystemExit(f"Unknown camera {camera!r} in {name}.")
    sequence, layout = describe_folder(folder)
    view = VIEWS[camera]
    video_id = f"{subject}_{sequence}_{layout}_{view}"
    sub = condition if scale is None else os.path.join(condition, f"s{scale:.1f}")
    target_dir = os.path.join(out_dir, sub)
    os.makedirs(target_dir, exist_ok=True)

    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    meta = document.get("_meta", {})
    rois = document.get("_rois", {})
    diagnostics = document.get("_diagnostics", {})
    trackers = [k for k in document if not k.startswith("_") and k not in REFERENCE_SECTIONS]
    references = [k for k in REFERENCE_SECTIONS if k in document]
    ratios = [r["w"] / r["d"] for r in rois.values() if r.get("d")]

    header = {
        "format": EXPORT_FORMAT,
        "video_id": video_id,
        "subject": subject,
        "sequence": sequence,
        "layout": layout,
        "view": view,
        "condition": condition,
        "box_scale": scale if scale is not None else (round(statistics.median(ratios), 2) if ratios else None),
        "frames": meta.get("frames"),
        "video_fps": 60,
        "frame_size": [1080, 1920],
        "opencv_version": meta.get("opencv_version"),
        "cv_num_threads": meta.get("cv_num_threads"),
        "reference_presets": meta.get("presets", {}),
        "reference_detector": meta.get("reference_detector", {}),
        "reference_streams": references,
        "trackers": trackers,
        "n_markers": len(rois),
        "rois": rois,
    }
    if "HybridVitCsrt" in diagnostics:
        hybrid = diagnostics["HybridVitCsrt"]
        header["hybrid"] = {"primary": hybrid.get("primary"), "fallback": hybrid.get("fallback")}
    with open(os.path.join(target_dir, video_id + ".json"), "w", encoding="utf-8") as handle:
        json.dump(header, handle, indent=1)

    rows = 0
    with gzip.open(os.path.join(target_dir, video_id + "_trackers.csv.gz"), "wt",
                   encoding="utf-8", newline="", compresslevel=9) as handle:
        writer = csv.writer(handle)
        writer.writerow(["tracker", "frame", "marker", "cx", "cy", "w", "h", "valid", "fallback"])
        for tracker in trackers:
            valid = diagnostics.get(tracker, {}).get("valid", {})
            switches = diagnostics.get(tracker, {}).get("fallback_switches", {})
            for key in frame_keys(document[tracker]):
                flags = valid.get(key, [])
                fallbacks = switches.get(key)
                for marker, (cx, cy, w, h) in enumerate(chunks(document[tracker][key])):
                    ok = flags[marker] if marker < len(flags) else True
                    fallback = "" if fallbacks is None else int(bool(fallbacks[marker]))
                    writer.writerow([tracker, int(key), marker, cx, cy, w, h, int(bool(ok)), fallback])
                    rows += 1
    with gzip.open(os.path.join(target_dir, video_id + "_reference.csv.gz"), "wt",
                   encoding="utf-8", newline="", compresslevel=9) as handle:
        writer = csv.writer(handle)
        writer.writerow(["stream", "frame", "x", "y"])
        for stream in references:
            for key in frame_keys(document[stream]):
                for x, y in (c[:2] for c in chunks(document[stream][key])):
                    writer.writerow([stream, int(key), x, y])

    return dict(video_id=video_id, condition=condition, box_scale=header["box_scale"],
                subject=subject, sequence=sequence, layout=layout, view=view,
                frames=header["frames"], n_markers=len(rois), trackers=len(trackers),
                file=os.path.join(sub, video_id + ".json").replace(os.sep, "/")), rows


def privacy_check(out_dir, codes):
    """Fails if a subject token, a recording folder name or a local path reached the data files.

    README.md and LICENSE are fixed texts (the licence names the authors), so only the path check
    applies to them.
    """
    tokens = [t.lower() for t in codes if t]
    paths = ["recordings", "/home/", "preview_", "users\\", "c:/"]
    for root, _, files in os.walk(out_dir):
        for name in files:
            path = os.path.join(root, name)
            opener = gzip.open if name.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
                text = handle.read().lower()
            forbidden = paths if name in ("README.md", "LICENSE") else paths + tokens
            for token in forbidden:
                if token in text or token in name.lower():
                    raise SystemExit(f"Privacy check failed: {token!r} found in {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="Output directory of run_benchmark.py.")
    parser.add_argument("--out", required=True, help="Dataset directory to write.")
    parser.add_argument("--codes", help="Subject code mapping (default: subject_codes.json here).")
    parser.add_argument("--annotations",
                        help="annotation_points.csv of analyse_annotations.py, copied into the dataset "
                             "(default: annotation/annotation_points.csv next to --results, if present).")
    args = parser.parse_args()

    codes = load_subject_codes(args.codes)
    if not codes:
        raise SystemExit("No subject code mapping found; the export never writes subject names.")
    files = results_files(args.results)
    if not files:
        raise SystemExit(f"No results found under {args.results}.")
    os.makedirs(args.out, exist_ok=True)

    index, total_rows = [], 0
    for condition, scale, path in files:
        entry, rows = export_one(condition, scale, path, args.out, codes)
        index.append(entry)
        total_rows += rows
        print(f"{entry['file']:<45} {entry['n_markers']:>3} markers {entry['trackers']:>2} trackers")
    with open(os.path.join(args.out, "videos.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index[0].keys()))
        writer.writeheader()
        writer.writerows(index)
    annotations = args.annotations or os.path.join(os.path.dirname(os.path.abspath(args.results)),
                                                   "annotation", "annotation_points.csv")
    if os.path.exists(annotations):
        # Coded video ids and anonymous annotator columns only; checked with everything else below.
        shutil.copyfile(annotations, os.path.join(args.out, "annotation_points.csv"))
        print(f"manual annotations: {annotations}")
    elif args.annotations:
        raise SystemExit(f"No annotations at {args.annotations}.")
    here = os.path.dirname(os.path.abspath(__file__))
    for template, target in (("DATASET_README.md", "README.md"), ("DATASET_LICENSE", "LICENSE")):
        with open(os.path.join(here, template), encoding="utf-8") as source, \
             open(os.path.join(args.out, target), "w", encoding="utf-8") as destination:
            destination.write(source.read())

    privacy_check(args.out, codes)
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(args.out) for f in fs)
    print(f"\n{len(index)} videos, {total_rows} tracker rows, {size / 1e6:.1f} MB written to {args.out}; "
          f"privacy check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
