"""Compares the manual annotations with the reference streams and the trackers.

  python analyse_annotations.py --kit ../Results/annotation --results ../Results/benchmark \\
                                annotations_AB_kit-....json annotations_CD_kit-....json
  python analyse_annotations.py --points data/annotation_points.csv --data data   # from the release

The manual centre of a marker in a frame is the mean of the annotators' clicks on its patch. It
shares nothing with the detectors, so it validates the reference independently. Reported:

  - inter-annotator difference (two annotators) and each annotator's repeatability (patches
    shown twice): these bound the precision of the manual reference itself;
  - Blob Detection and PCWE against the manual centres: mean and median distance, bias in x and
    y, Bland-Altman 95 % limits of agreement, gross errors (more than half a marker diameter
    away: another blob), and how often a stream has no position for a marker the annotators saw
    or a position for one they could not see - overall, per layout, per view and per stratum;
  - every tracker's mean distance from the manual centres next to its mean distance from the
    Blob reference on the same marker-frames (frames where the tracker reports the marker as
    tracked, as in the main analysis), and Kendall's tau between the two orderings: does the
    ranking of the trackers hold against an independent reference?

Writes <kit>/annotation_points.csv (one row per annotated marker-frame, coded video ids only, the
part of the study that can be released) and <kit>/annotation_summary.json.

Standard library only.
"""

import argparse
import csv
import glob
import gzip
import json
import math
import os
import statistics

from analysis_common import (REFERENCE_SECTIONS, TRACKER_NAMES, build_reference_tracks, frame_keys,
                             load_results, parse_points, roi_centres)


def summary(differences):
    """Distances and signed differences (dx, dy) -> agreement statistics."""
    if not differences:
        return {"n": 0}
    distances = [math.hypot(dx, dy) for dx, dy in differences]
    out = {"n": len(differences), "mean": statistics.mean(distances), "median": statistics.median(distances),
           "p95": sorted(distances)[min(len(distances) - 1, int(0.95 * len(distances)))], "max": max(distances)}
    for axis, values in (("x", [d[0] for d in differences]), ("y", [d[1] for d in differences])):
        bias = statistics.mean(values)
        spread = statistics.stdev(values) if len(values) > 1 else 0.0
        out[f"bias_{axis}"] = bias
        out[f"loa_{axis}"] = [bias - 1.96 * spread, bias + 1.96 * spread]
    return out


def kendall_tau(a, b):
    concordant = discordant = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            s = (a[i] - a[j]) * (b[i] - b[j])
            concordant += s > 0
            discordant += s < 0
    pairs = len(a) * (len(a) - 1) / 2
    return (concordant - discordant) / pairs if pairs else float("nan")


def line(label, s):
    if not s.get("n"):
        return f"  {label:<26} n=0"
    return (f"  {label:<26} n={s['n']:<4} mean {s['mean']:5.2f}  median {s['median']:5.2f}  p95 {s['p95']:5.2f}  "
            f"bias ({s['bias_x']:+.2f}, {s['bias_y']:+.2f})  LoA x [{s['loa_x'][0]:+.2f}, {s['loa_x'][1]:+.2f}]"
            f" y [{s['loa_y'][0]:+.2f}, {s['loa_y'][1]:+.2f}]")


def summarise_points(path, data_dir=None):
    """Agreement statistics from annotation_points.csv alone, as released with the dataset. The
    manual centre is the mean of the two annotators where both marked the marker; the 'start' and
    'peak' rows are the representative sample, the 'disagreement' rows a selection of hard cases."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    value = lambda r, k: float(r[k]) if r[k] != "" else None
    for title, keep in (("start and peak frames (representative)", ("start", "peak")),
                        ("disagreement frames (selected hard cases)", ("disagreement",))):
        print(f"\n{title}:")
        for layout in (None, "4", "40"):
            sel = [r for r in rows if r["stratum"] in keep and (layout is None or r["layout"] == layout)]
            inter, streams, missed = [], {"blob": [], "pcwe": []}, {"blob": 0, "pcwe": 0}
            for r in sel:
                a1, a2 = (value(r, "annotator_1_x"), value(r, "annotator_1_y")), (value(r, "annotator_2_x"), value(r, "annotator_2_y"))
                if None in a1 or None in a2:
                    continue
                inter.append((a1[0] - a2[0], a1[1] - a2[1]))
                centre = ((a1[0] + a2[0]) / 2, (a1[1] + a2[1]) / 2)
                for stream in streams:
                    s = (value(r, f"{stream}_x"), value(r, f"{stream}_y"))
                    if None in s:
                        missed[stream] += 1
                    else:
                        streams[stream].append((s[0] - centre[0], s[1] - centre[1]))
            tag = "all layouts" if layout is None else f"layout={layout}"
            print(line(f"annotators, {tag}", summary(inter)))
            for stream, name in (("blob", "Blob Detection"), ("pcwe", "PCWE")):
                seen = len(streams[stream]) + missed[stream]
                print(line(f"{name}, {tag}", summary(streams[stream]))
                      + f"  missed {missed[stream]} ({100 * missed[stream] / max(1, seen):.1f}%)")
    if data_dir:
        compare_trackers(rows, data_dir)
    return 0


def compare_trackers(rows, data_dir):
    """Every tracker's mean distance from the manual centres and from the Blob reference on the same
    annotated marker-frames (all rows; frames where the tracker reports the marker as tracked)."""
    value = lambda r, k: float(r[k]) if r[k] != "" else None
    wanted = {}
    for r in rows:
        a1, a2 = (value(r, "annotator_1_x"), value(r, "annotator_1_y")), (value(r, "annotator_2_x"), value(r, "annotator_2_y"))
        blob = (value(r, "blob_x"), value(r, "blob_y"))
        if None in a1 or None in a2 or None in blob:
            continue
        manual = ((a1[0] + a2[0]) / 2, (a1[1] + a2[1]) / 2)
        wanted.setdefault(r["video_id"], {})[(int(r["frame"]), int(r["marker"]))] = (manual, blob)
    distances = {}
    for video_id, points in wanted.items():
        found = glob.glob(os.path.join(data_dir, "**", f"{video_id}_trackers.csv.gz"), recursive=True)
        found = [f for f in found if os.sep + "sweep" + os.sep not in f]
        if len(found) != 1:
            raise SystemExit(f"{video_id}: expected one tracker file under {data_dir}, found {len(found)}")
        with gzip.open(found[0], "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (int(row["frame"]), int(row["marker"]))
                if key in points and row["valid"] == "1":
                    manual, blob = points[key]
                    position = (float(row["cx"]), float(row["cy"]))
                    name = TRACKER_NAMES.get(row["tracker"], row["tracker"])
                    distances.setdefault(name, []).append((math.dist(position, manual), math.dist(position, blob)))
    order = sorted(distances)
    tau = kendall_tau([statistics.mean(d[0] for d in distances[t]) for t in order],
                      [statistics.mean(d[1] for d in distances[t]) for t in order])
    print("\nTrackers: mean distance from the manual centre and from the Blob reference (px), same marker-frames:")
    for name in sorted(distances, key=lambda t: statistics.mean(d[0] for d in distances[t])):
        d = distances[name]
        print(f"  {name:<20} n={len(d):<4} manual {statistics.mean(x[0] for x in d):7.2f}   Blob {statistics.mean(x[1] for x in d):7.2f}")
    print(f"  Kendall tau between the two orderings: {tau:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kit", help="directory with manifest.json")
    parser.add_argument("--results", help="run_benchmark.py output directory")
    parser.add_argument("annotations", nargs="*", help="files exported from the annotation page")
    parser.add_argument("--points", help="summarise a released annotation_points.csv instead (no kit or "
                                         "results needed); repeatability needs the kit")
    parser.add_argument("--data", help="with --points: the released dataset directory, to compare the "
                                       "trackers with the manual centres and the reference")
    args = parser.parse_args()
    if args.points:
        return summarise_points(args.points, args.data)
    if not (args.kit and args.results and args.annotations):
        parser.error("--kit, --results and the exported annotation files are required (or --points)")

    with open(os.path.join(args.kit, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    patches = {p["id"]: p for p in manifest["patches"]}
    annotators = {}
    for path in args.annotations:
        with open(path, encoding="utf-8") as handle:
            export = json.load(handle)
        if export["kit"] != manifest["kit"]:
            raise SystemExit(f"{path} belongs to {export['kit']}, the manifest to {manifest['kit']}.")
        annotators[export["annotator"]] = {a["id"]: a for a in export["answers"]}
    names = sorted(annotators)

    def frame_point(patch_id, answers):
        """Frame coordinates of an annotator's click, None for 'not visible', missing if unanswered."""
        answer = answers.get(patch_id)
        if answer is None:
            return "missing"
        if answer["status"] != "clicked":
            return None
        patch = patches[patch_id]
        return (patch["x0"] + answer["x"], patch["y0"] + answer["y"])

    # Repeatability: the same marker-frame on two patches cut differently.
    repeatability = {}
    for name in names:
        differences = []
        for patch in patches.values():
            if patch.get("repeat"):
                a, b = frame_point(patch["repeat_of"], annotators[name]), frame_point(patch["id"], annotators[name])
                if isinstance(a, tuple) and isinstance(b, tuple):
                    differences.append((b[0] - a[0], b[1] - a[1]))
        repeatability[name] = summary(differences)

    # One record per annotated marker-frame (the first patch of it).
    documents, tracks = {}, {}
    records = []
    for patch in patches.values():
        if patch.get("repeat"):
            continue
        clicks = {name: frame_point(patch["id"], annotators[name]) for name in names}
        if any(c == "missing" for c in clicks.values()):
            continue
        path = os.path.join(args.results, patch["results"])
        if path not in documents:
            documents[path] = load_results(path)
            seeds = roi_centres(documents[path])
            tracks[path] = {}
            for section in REFERENCE_SECTIONS:
                detections = {k: parse_points(documents[path][section][k]) for k in frame_keys(documents[path][section])}
                tracks[path][section] = build_reference_tracks(detections, seeds)
        document, key, marker = documents[path], f"{patch['frame']:04d}", patch["marker"]
        visible = [c for c in clicks.values() if c is not None]
        record = dict(video_id=patch["video_id"], layout=patch["layout"], view=patch["view"], frame=patch["frame"],
                      marker=marker, stratum=patch["stratum"], diameter=document["_rois"][str(marker)]["d"],
                      clicks=clicks, manual=None, agreement=len(visible) == len(clicks))
        if visible and record["agreement"]:
            record["manual"] = (statistics.mean(c[0] for c in visible), statistics.mean(c[1] for c in visible))
        for section in REFERENCE_SECTIONS:
            record[section] = tracks[path][section].get(key, [None] * (marker + 1))[marker]
        # Tracker positions in frames where the tracker reports the marker as tracked (active), as
        # in the main analysis; after a reported loss the box no longer marks anything.
        record["trackers"] = {}
        for section in document:
            if section in TRACKER_NAMES:
                boxes = parse_points(document[section].get(key, ""))
                flags = document.get("_diagnostics", {}).get(section, {}).get("valid", {}).get(key)
                if marker < len(boxes) and (flags is None or flags[marker]):
                    record["trackers"][TRACKER_NAMES[section]] = boxes[marker][:2]
        records.append(record)

    # Inter-annotator agreement on marker-frames that all annotators clicked.
    inter = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            differences = [(r["clicks"][b][0] - r["clicks"][a][0], r["clicks"][b][1] - r["clicks"][a][1])
                           for r in records if isinstance(r["clicks"][a], tuple) and isinstance(r["clicks"][b], tuple)]
            visibility = sum((r["clicks"][a] is None) != (r["clicks"][b] is None) for r in records)
            inter[f"{a} vs {b}"] = dict(summary(differences), visibility_disagreements=visibility)

    labelled = [r for r in records if r["manual"] is not None]
    unseen = [r for r in records if all(c is None for c in r["clicks"].values())]
    streams = {}
    for section, label in (("MultiBlobDetector", "Blob Detection"), ("WeightedEdgesExtractor", "PCWE")):
        def compare(subset):
            present = [r for r in subset if r[section] is not None]
            differences = [(r[section][0] - r["manual"][0], r[section][1] - r["manual"][1]) for r in present]
            gross = sum(math.hypot(*d) > 0.5 * r["diameter"] for d, r in zip(differences, present))
            out = summary(differences)
            out.update(missed=len(subset) - len(present), gross=gross)
            return out
        groups = {"all": labelled}
        for field in ("layout", "view", "stratum"):
            for value in sorted({r[field] for r in labelled}):
                groups[f"{field}={value}"] = [r for r in labelled if r[field] == value]
        streams[label] = {group: compare(subset) for group, subset in groups.items()}
        streams[label]["positions_where_annotators_saw_no_marker"] = sum(r[section] is not None for r in unseen)

    # Trackers against the manual centres and against the Blob reference, on the same marker-frames.
    trackers = {}
    for name in sorted({t for r in labelled for t in r["trackers"]}):
        both = [r for r in labelled if name in r["trackers"] and r["MultiBlobDetector"] is not None]
        if not both:
            continue
        manual = statistics.mean(math.dist(r["trackers"][name], r["manual"]) for r in both)
        blob = statistics.mean(math.dist(r["trackers"][name], r["MultiBlobDetector"]) for r in both)
        trackers[name] = {"n": len(both), "vs_manual": manual, "vs_blob": blob}
    order = sorted(trackers)
    tau = kendall_tau([trackers[t]["vs_manual"] for t in order], [trackers[t]["vs_blob"] for t in order])

    print(f"{len(records)} marker-frames annotated by {', '.join(names)}; {len(labelled)} seen by all, "
          f"{len(unseen)} by none, {len(records) - len(labelled) - len(unseen)} with disagreeing visibility\n")
    print("Manual reference precision (px):")
    for pair, s in inter.items():
        print(line(f"inter-annotator {pair}", s) + f"  visibility disagreements {s['visibility_disagreements']}")
    for name, s in repeatability.items():
        print(line(f"repeatability {name}", s))
    for label, groups in streams.items():
        print(f"\n{label} vs manual centre (px):")
        for group, s in groups.items():
            if isinstance(s, dict):
                print(line(group, s) + (f"  missed {s['missed']}  gross {s['gross']}" if s.get("n") is not None else ""))
        print(f"  positions where the annotators saw no marker: {groups['positions_where_annotators_saw_no_marker']} "
              f"of {len(unseen)}")
    print("\nTrackers: mean distance from the manual centre and from the Blob reference (px), same marker-frames:")
    for name in sorted(trackers, key=lambda t: trackers[t]["vs_manual"]):
        t = trackers[name]
        print(f"  {name:<20} n={t['n']:<4} manual {t['vs_manual']:7.2f}   Blob {t['vs_blob']:7.2f}")
    print(f"  Kendall tau between the two orderings: {tau:.3f}")

    with open(os.path.join(args.kit, "annotation_points.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # Annotators appear as annotator_1, annotator_2, ... (in the order of their initials).
        writer.writerow(["video_id", "frame", "marker", "stratum", "layout", "view"] +
                        [f"annotator_{i}_{axis}" for i in range(1, len(names) + 1) for axis in ("x", "y")] +
                        ["blob_x", "blob_y", "pcwe_x", "pcwe_y"])
        for r in sorted(records, key=lambda r: (r["video_id"], r["frame"], r["marker"])):
            row = [r["video_id"], r["frame"], r["marker"], r["stratum"], r["layout"], r["view"]]
            for name in names:
                c = r["clicks"][name]
                row += [f"{c[0]:.2f}", f"{c[1]:.2f}"] if c else ["", ""]
            for section in REFERENCE_SECTIONS:
                p = r[section]
                row += [f"{p[0]:.3f}", f"{p[1]:.3f}"] if p else ["", ""]
            writer.writerow(row)
    with open(os.path.join(args.kit, "annotation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"inter_annotator": inter, "repeatability": repeatability, "streams": streams,
                   "trackers": trackers, "kendall_tau": tau}, handle, indent=1)
    print(f"\nwrote {args.kit}/annotation_points.csv and annotation_summary.json")


if __name__ == "__main__":
    main()
