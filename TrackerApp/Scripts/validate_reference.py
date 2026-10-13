"""Compares the Blob Detection and PCWE reference streams, over all markers and without gating.

Every marker is followed through the sequence, with separate candidate searches for the two
streams. Every frame is kept: frames in which the two methods disagree by more than the gate are
marked rather than dropped, because the upper tail of the disagreement distribution is what a
validation has to measure. The agreement statistics are reported both including and excluding
the marked frames, so the effect of the gate is visible. Coordinates are parsed as float.

Usage:
  python3 validate_reference.py --results Results/run.json --out-csv reference_agreement.csv
"""

import argparse
import csv
import math
import os
import statistics

from analysis_common import build_reference_tracks, frame_keys, load_results, parse_points, roi_centres


def collect(document, section):
    data = document.get(section)
    if data is None:
        return {}
    return {key: parse_points(data[key]) for key in frame_keys(data)}


def describe(values, label):
    if not values:
        return f"  {label:<28} n=0"
    return (f"  {label:<28} n={len(values):<6} mean={statistics.mean(values):7.4f}  "
            f"sd={statistics.pstdev(values) if len(values) > 1 else 0.0:7.4f}  "
            f"median={statistics.median(values):7.4f}  max={max(values):7.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, nargs="+")
    parser.add_argument("--out-csv", default="reference_agreement.csv")
    parser.add_argument("--max-jump", type=float, default=25.0,
                        help="Largest accepted frame-to-frame motion when following a marker.")
    parser.add_argument("--gate-px", type=float, default=10.0,
                        help="Disagreement above which a frame is MARKED gated. Frames are never "
                             "dropped; this only labels them so the gate's effect is measurable.")
    args = parser.parse_args()

    rows = []
    for path in args.results:
        document = load_results(path)
        # The coded id of an exported video, else the results file name; never the stored video
        # path, whose folder name identifies the participant.
        sequence = document.get("_meta", {}).get("video_id") or os.path.basename(path)[:-len(".json")]
        seeds = roi_centres(document)
        if not seeds:
            raise SystemExit(f"{path}: no \"_rois\" section; cannot seed marker tracks.")

        blob = collect(document, "MultiBlobDetector")
        edges = collect(document, "WeightedEdgesExtractor")
        if not blob or not edges:
            raise SystemExit(f"{path}: needs both MultiBlobDetector and WeightedEdgesExtractor "
                             f"sections. Run TrackerBench --trackers BlobDetector,Reference")

        # Each stream is followed independently, from the same seeds. Deriving one from the other
        # would reintroduce exactly the dependence this comparison is meant to test.
        blob_tracks = build_reference_tracks(blob, seeds, max_jump_px=args.max_jump)
        edge_tracks = build_reference_tracks(edges, seeds, max_jump_px=args.max_jump)

        for frame in sorted(set(blob_tracks) | set(edge_tracks)):
            blob_frame = blob_tracks.get(frame, [])
            edge_frame = edge_tracks.get(frame, [])
            for marker in range(len(seeds)):
                b = blob_frame[marker] if marker < len(blob_frame) else None
                e = edge_frame[marker] if marker < len(edge_frame) else None
                agreement = math.dist(b, e) if (b and e) else None
                rows.append(dict(
                    sequence=sequence, marker_id=marker, frame=int(frame),
                    blob_x=("" if b is None else f"{b[0]:.3f}"),
                    blob_y=("" if b is None else f"{b[1]:.3f}"),
                    edges_x=("" if e is None else f"{e[0]:.3f}"),
                    edges_y=("" if e is None else f"{e[1]:.3f}"),
                    blob_present=int(b is not None), edges_present=int(e is not None),
                    agreement_px=("" if agreement is None else f"{agreement:.4f}"),
                    gated=("" if agreement is None else int(agreement > args.gate_px))))

    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    paired = [float(r["agreement_px"]) for r in rows if r["agreement_px"] != ""]
    ungated = [float(r["agreement_px"]) for r in rows if r["agreement_px"] != "" and r["gated"] == 0]
    both_missing = sum(1 for r in rows if r["agreement_px"] == "")

    print(f"{len(rows)} marker-frames -> {args.out_csv}")
    print(f"markers followed: {len(set(r['marker_id'] for r in rows))}   "
          f"frames with no paired detection: {both_missing}\n")
    print("Blob Detection vs PCWE agreement (px):")
    print(describe(paired, "all paired frames"))
    print(describe(ungated, f"excluding >{args.gate_px:g} px (gated)"))
    if paired and ungated and len(paired) != len(ungated):
        dropped = len(paired) - len(ungated)
        print(f"\n  The gate marks {dropped} of {len(paired)} paired frames "
              f"({100 * dropped / len(paired):.1f}%), all from the upper tail;")
        print(f"  excluding them lowers the mean by "
              f"{statistics.mean(paired) - statistics.mean(ungated):.4f} px.")


if __name__ == "__main__":
    main()
