"""Computes per-frame tracking deviations from the reference and the summary metrics.

A tracker has lost its marker in a frame when it reports failure, when it sits nearer another
marker than its own (identity swap), or when its centre is more than two marker diameters from
its marker (drift onto another structure; --lost-beyond, see analysis_common.tracking_state).
Lost frames are excluded from the conditional deviation and never imputed; the per-frame CSV
records the state of every frame. This script computes only. Standard library only.

Metrics produced, and what each answers:

  conditional deviation   Mean deviation over frames where tracking was active. It is explicitly
                          conditional, and the exposure column says what fraction of frames it
                          covers; lost_reported, lost_swap and lost_drift count why the others
                          were lost.
  penalised deviation     Mean over every frame, with lost frames charged a fixed penalty. Ranks
                          trackers without the survivor bias of the conditional figure.
  signed bias             Mean signed dx and dy, which exposes systematic offsets that absolute
                          component deviations hide.
  mis-assignment rate     Fraction of the frames the tracker reported as tracked in which it sat
                          closer to a different reference marker than to its own - an objective
                          identity-swap detector.
  retention area          Mean fraction of markers still valid across the sequence, so a tracker
                          that fails early scores worse than one that fails at the end.
  median survival         Median frame index at which a marker was last continuously valid.

Usage:
  python3 analyse_tracking.py --results Results/run.json --out-csv per_frame.csv

After the per-file summaries it prints one table pooled over every frame of all the files given, per
tracker: mean +- SD of |dx|, |dy| and the distance over active frames, their maxima, and the share
of scored frames in which the tracker was active (Table 4 of the article for the 4-marker files).
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys

from analysis_common import (LOST_BEYOND_DIAMETERS, REFERENCE_SECTIONS, build_reference_tracks,
                             frame_keys, load_results, marker_diameters, parse_points, roi_centres,
                             tracker_names, tracking_state)


def collect_detections(document, section_name):
    """Detections per frame from a reference section."""
    section = document.get(section_name)
    if section is None:
        return {}
    return {key: parse_points(section[key]) for key in frame_keys(section)}


def validity_for(document, tracker, frame, target_count):
    """Per-target validity flags for one frame.

    Uses the flags recorded under "_diagnostics". Results files without that section carry no
    validity information; for those the caller is warned and every target is treated as active.
    """
    diagnostics = document.get("_diagnostics", {}).get(tracker, {})
    flags = diagnostics.get("valid", {}).get(frame)
    if flags is None:
        return [True] * target_count
    return [bool(f) for f in flags] + [True] * (target_count - len(flags))


def analyse(document, sequence_id, reference_section, max_jump, penalty_px, lost_beyond=LOST_BEYOND_DIAMETERS):
    """Builds the per-frame record list and the per-tracker summaries."""
    seeds = roi_centres(document)
    if not seeds:
        raise SystemExit(
            "This results file has no \"_rois\" section, so reference tracks cannot be seeded\n"
            "independently of the trackers. Re-run with the current TrackerApp/TrackerBench,\n"
            "which persists the initial ROIs.")

    detections = collect_detections(document, reference_section)
    if not detections:
        raise SystemExit(f"No \"{reference_section}\" section in the results file. "
                         f"Run the reference tracker, or pass --reference.")

    reference = build_reference_tracks(detections, seeds, max_jump_px=max_jump)
    diameters = marker_diameters(document)
    roi_diagonals = []
    for entry in sorted(document["_rois"].items(), key=lambda kv: int(kv[0])):
        roi = entry[1]
        roi_diagonals.append(math.hypot(roi["w"], roi["h"]))

    rows = []
    summaries = []
    trackers = [t for t in tracker_names(document) if t not in REFERENCE_SECTIONS]

    for tracker in trackers:
        section = document[tracker]
        keys = frame_keys(section)
        if not keys:
            continue

        errors_active, dx_active, dy_active = [], [], []
        penalised, mis_assigned, active_count, total_count = [], 0, 0, 0
        scoreable_count = 0      # observations where a reference position exists
        reported_valid_count = 0 # ... of which the tracker reported as tracked
        lost_count = {"reported": 0, "swap": 0, "drift": 0}
        per_frame_valid_fraction = []
        last_valid_frame = [None] * len(seeds)
        frame_index = {key: i for i, key in enumerate(keys)}

        for key in keys:
            targets = parse_points(section[key])
            flags = validity_for(document, tracker, key, len(targets))
            reference_frame = reference.get(key, [None] * len(seeds))
            valid_here = 0

            for marker in range(min(len(targets), len(seeds))):
                total_count += 1
                target = targets[marker]
                is_valid = flags[marker] if marker < len(flags) else True
                reference_point = reference_frame[marker] if marker < len(reference_frame) else None

                state, error, dx, dy, nearest_marker = tracking_state(
                    target, marker, is_valid, reference_frame, diameters[marker], lost_beyond)

                # Still holding its marker as far as can be told: reported as tracked, and not
                # found on another marker or off the marker where the reference sees it.
                if is_valid and state not in ("swap", "drift"):
                    valid_here += 1
                    last_valid_frame[marker] = frame_index[key]

                if state == "unscored":
                    # No reference position for this marker in this frame, so the observation
                    # cannot be scored either way. Excluded from both metrics: an absent
                    # reference is a limitation of the detector, not a tracker failure, and
                    # charging the tracker a penalty for it would be wrong.
                    pass
                elif state == "active":
                    scoreable_count += 1
                    reported_valid_count += 1
                    errors_active.append(error)
                    dx_active.append(dx)
                    dy_active.append(dy)
                    penalised.append(error)
                    active_count += 1
                else:
                    # Lost: reported failure, identity swap or drift off the marker. Never
                    # impute: the observation is excluded from the conditional metric and
                    # charged a penalty in the all-frames metric.
                    scoreable_count += 1
                    reported_valid_count += state != "reported"
                    lost_count[state] += 1
                    mis_assigned += state == "swap"
                    penalised.append(penalty_px if penalty_px is not None
                                     else roi_diagonals[marker])

                rows.append(dict(
                    sequence=sequence_id, tracker=tracker, marker_id=marker, frame=int(key),
                    tx=f"{target[0]:.3f}", ty=f"{target[1]:.3f}",
                    rx=("" if reference_point is None else f"{reference_point[0]:.3f}"),
                    ry=("" if reference_point is None else f"{reference_point[1]:.3f}"),
                    error_px=("" if error is None else f"{error:.4f}"),
                    dx=("" if dx is None else f"{dx:.4f}"),
                    dy=("" if dy is None else f"{dy:.4f}"),
                    valid=int(bool(is_valid)),
                    reference_present=int(reference_point is not None),
                    state=state,
                    mis_assigned=("" if error is None else int(state == "swap")),
                    nearest_marker=("" if nearest_marker is None else nearest_marker)))

            per_frame_valid_fraction.append(valid_here / max(1, min(len(targets), len(seeds))))

        survivals = [(v + 1) if v is not None else 0 for v in last_valid_frame]
        diagnostics = document.get("_diagnostics", {}).get(tracker, {})
        frame_times = diagnostics.get("frame_times_us", [])

        summaries.append(dict(
            sequence=sequence_id,
            tracker=tracker,
            n_targets=diagnostics.get("n_targets", len(seeds)),
            frames=len(keys),
            observations=total_count,
            scoreable=scoreable_count,
            active=active_count,
            reference_coverage=scoreable_count / total_count if total_count else 0.0,
            exposure=active_count / scoreable_count if scoreable_count else 0.0,
            mean_error_px=statistics.mean(errors_active) if errors_active else float("nan"),
            sd_error_px=statistics.pstdev(errors_active) if len(errors_active) > 1 else 0.0,
            median_error_px=statistics.median(errors_active) if errors_active else float("nan"),
            mean_abs_dx=statistics.mean(abs(v) for v in dx_active) if dx_active else float("nan"),
            mean_abs_dy=statistics.mean(abs(v) for v in dy_active) if dy_active else float("nan"),
            bias_dx=statistics.mean(dx_active) if dx_active else float("nan"),
            bias_dy=statistics.mean(dy_active) if dy_active else float("nan"),
            penalised_error_px=statistics.mean(penalised) if penalised else float("nan"),
            mis_assignment_rate=mis_assigned / reported_valid_count if reported_valid_count else 0.0,
            lost_reported=lost_count["reported"],
            lost_swap=lost_count["swap"],
            lost_drift=lost_count["drift"],
            retention_area=(statistics.mean(per_frame_valid_fraction)
                            if per_frame_valid_fraction else 0.0),
            retention_final=(per_frame_valid_fraction[-1] if per_frame_valid_fraction else 0.0),
            median_survival_frames=statistics.median(survivals),
            mean_frame_time_ms=(statistics.mean(frame_times) / 1000.0 if frame_times else ""),
            sd_frame_time_ms=(statistics.pstdev(frame_times) / 1000.0
                              if len(frame_times) > 1 else "")))

    return rows, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, nargs="+",
                        help="Results JSON file(s) from TrackerApp or TrackerBench.")
    parser.add_argument("--out-csv", default="per_frame_errors.csv",
                        help="Per-frame deviation table (the released artefact).")
    parser.add_argument("--out-summary", default="tracker_summary.csv",
                        help="Per-tracker summary metrics.")
    parser.add_argument("--reference", default="MultiBlobDetector", choices=REFERENCE_SECTIONS,
                        help="Section used as the reference trajectory.")
    parser.add_argument("--max-jump", type=float, default=25.0,
                        help="Largest accepted frame-to-frame reference motion, in pixels.")
    parser.add_argument("--penalty-px", type=float, default=None,
                        help="Deviation charged for a lost frame in the penalised metric. "
                             "Defaults to each marker's ROI diagonal.")
    parser.add_argument("--lost-beyond", type=float, default=LOST_BEYOND_DIAMETERS,
                        help="A tracker more than this many marker diameters from its marker has "
                             "lost it (default %(default)s; 0 switches the condition off).")
    args = parser.parse_args()

    all_rows, all_summaries = [], []
    for path in args.results:
        document = load_results(path)
        # The coded id of an exported video, else the results file name; never the stored video
        # path, whose folder name identifies the participant.
        sequence_id = document.get("_meta", {}).get("video_id") or os.path.basename(path)[:-len(".json")]
        if "_diagnostics" not in document:
            print(f"warning: {path} has no \"_diagnostics\" section (older results file); "
                  f"validity flags unavailable, all frames treated as active.", file=sys.stderr)
        rows, summaries = analyse(document, sequence_id, args.reference,
                                  args.max_jump, args.penalty_px, args.lost_beyond)
        all_rows.extend(rows)
        all_summaries.extend(summaries)

    if not all_rows:
        raise SystemExit("No tracker data found in the supplied results files.")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with open(args.out_summary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        for row in all_summaries:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})

    print(f"{len(all_rows)} per-frame records -> {args.out_csv}")
    print(f"{len(all_summaries)} tracker summaries -> {args.out_summary}\n")
    header = (f"{'tracker':<18}{'refcov':>8}{'expo':>7}{'mean_px':>9}{'pen_px':>8}"
              f"{'bias_dx':>9}{'bias_dy':>9}{'misasgn':>9}{'ret_area':>9}")
    print(header)
    print("-" * len(header))
    def sort_key(row):
        value = row["mean_error_px"]
        return (1, 0.0) if value != value else (0, value)      # NaN (no active frames) last

    for row in sorted(all_summaries, key=sort_key):
        print(f"{row['tracker']:<18}{row['reference_coverage']:>8.2f}{row['exposure']:>7.2f}"
              f"{row['mean_error_px']:>9.3f}"
              f"{row['penalised_error_px']:>8.2f}{row['bias_dx']:>9.3f}{row['bias_dy']:>9.3f}"
              f"{row['mis_assignment_rate']:>9.3f}{row['retention_area']:>9.3f}")


    pooled_table(all_rows)


def pooled_table(rows):
    """Per tracker, pooled over every active frame of all files: absolute component deviations,
    distance (mean +- population SD, maximum) and the share of scored frames that were active."""
    pooled = {}
    for row in rows:
        if row["state"] == "unscored":
            continue
        entry = pooled.setdefault(row["tracker"], {"dx": [], "dy": [], "d": [], "traj": set(), "scored": 0})
        entry["scored"] += 1
        if row["state"] != "active":
            continue
        entry["dx"].append(abs(float(row["dx"])))
        entry["dy"].append(abs(float(row["dy"])))
        entry["d"].append(float(row["error_px"]))
        entry["traj"].add((row["sequence"], row["marker_id"]))

    def ms(values):
        return f"{statistics.mean(values):5.2f}+-{statistics.pstdev(values):4.2f}"

    print("\nPooled over all frames of all files (active frames only):")
    header = (f"{'tracker':<18}{'traj':>5}{'frames':>8}  {'|dx|':>10}{'max':>7}  {'|dy|':>10}{'max':>7}  "
              f"{'distance':>10}{'max':>7}{'active %':>10}")
    print(header)
    print("-" * len(header))
    for name, e in sorted(((n, e) for n, e in pooled.items() if e["d"]), key=lambda item: statistics.mean(item[1]["d"])):
        print(f"{name:<18}{len(e['traj']):>5}{len(e['d']):>8}  {ms(e['dx'])}{max(e['dx']):7.2f}  "
              f"{ms(e['dy'])}{max(e['dy']):7.2f}  {ms(e['d'])}{max(e['d']):7.2f}{100 * len(e['d']) / e['scored']:10.1f}")


if __name__ == "__main__":
    main()
