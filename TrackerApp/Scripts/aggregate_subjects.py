"""Aggregates benchmark results to the subject level and summarises them per tracker.

* Deviations are aggregated bottom-up. Per-frame observations are pooled to a trajectory (one
  marker in one camera view of one sequence), trajectories to a subject, and only then across
  subjects. Frames are autocorrelated and nested within markers, views, sequences and subjects,
  so the pooled per-frame standard deviation badly understates uncertainty. The independent
  experimental unit is the subject.
* Uncertainty is reported as a 95% percentile bootstrap confidence interval over subjects. The
  printed grouping of trackers by overlapping intervals is descriptive only; overlap of intervals
  is not a test of a difference.
* Both the conditional deviation (active frames only, with an exposure column) and the penalised
  deviation (all scoreable frames, failures charged a penalty) are reported, because they can
  rank trackers differently: a tracker that fails early can show a small conditional deviation
  computed over few frames. A frame is lost when the tracker reports failure, swaps onto another
  marker, or drifts more than two marker diameters off its marker (--lost-beyond; see
  analysis_common.tracking_state).

Standard library only. Writes a per-trajectory CSV and a JSON summary.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import statistics
from collections import defaultdict

from analysis_common import (LOST_BEYOND_DIAMETERS, REFERENCE_SECTIONS, TRACKER_NAMES, VIEWS,
                             build_reference_tracks, frame_keys, load_results, load_subject_codes,
                             marker_diameters, parse_points, roi_centres, subject_of, tracking_state)

MM_PER_PIXEL = 0.24

SUBJECT_CODES = load_subject_codes()

PRETTY = TRACKER_NAMES


def bootstrap_ci(values, iterations=10000, alpha=0.05, seed=20260907):
    """Percentile bootstrap CI of the mean over independent units (subjects)."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(iterations):
        means.append(sum(rng.choice(values) for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * iterations)]
    hi = means[int((1 - alpha / 2) * iterations) - 1]
    return (lo, hi)


def sign_flip_p(a, b, iterations=10000, seed=7):
    """Two-sided p for a paired difference over subjects, by Monte Carlo sign-flip permutation."""
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs:
        return float("nan")
    observed = abs(sum(diffs) / len(diffs))
    rng = random.Random(seed)
    count = 0
    for _ in range(iterations):
        flipped = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(sum(flipped) / len(flipped)) >= observed - 1e-15:
            count += 1
    return count / iterations


def analyse_file(path, max_jump, penalty_px, lost_beyond=LOST_BEYOND_DIAMETERS):
    """Per-trajectory statistics for one results file.

    \returns list of dicts, one per (tracker, marker).
    """
    document = load_results(path)
    seeds = roi_centres(document)
    if not seeds:
        return []
    blob = document.get("MultiBlobDetector")
    if blob is None:
        return []
    detections = {k: parse_points(blob[k]) for k in frame_keys(blob)}
    reference = build_reference_tracks(detections, seeds, max_jump_px=max_jump)

    diameters = marker_diameters(document)
    roi_diag = []
    for _, roi in sorted(document["_rois"].items(), key=lambda kv: int(kv[0])):
        roi_diag.append(math.hypot(roi["w"], roi["h"]))

    base = os.path.basename(path).replace(".json", "")
    sequence, camera = base.split("__") if "__" in base else (base, "")
    rows = []

    for tracker in [t for t in document if not t.startswith("_") and t not in REFERENCE_SECTIONS]:
        section = document[tracker]
        keys = frame_keys(section)
        if not keys:
            continue
        diagnostics = document.get("_diagnostics", {}).get(tracker, {})
        validity = diagnostics.get("valid", {})

        per_marker = defaultdict(lambda: dict(err=[], adx=[], ady=[], sdx=[], sdy=[],
                                              pen=[], scoreable=0, active=0, mis=0, reported_valid=0,
                                              last_valid=-1, valid_frames=0))
        for index, key in enumerate(keys):
            targets = parse_points(section[key])
            flags = validity.get(key, [True] * len(targets))
            ref_frame = reference.get(key, [None] * len(seeds))
            for m in range(min(len(targets), len(seeds))):
                rec = per_marker[m]
                ok = bool(flags[m]) if m < len(flags) else True
                state, d, dx, dy, _ = tracking_state(targets[m], m, ok, ref_frame, diameters[m], lost_beyond)
                if ok and state not in ("swap", "drift"):
                    rec["valid_frames"] += 1
                    rec["last_valid"] = index
                if state == "unscored":
                    continue
                rec["scoreable"] += 1
                rec["reported_valid"] += state != "reported"
                rec["mis"] += state == "swap"
                if state != "active":       # lost: reported failure, swap or drift
                    rec["pen"].append(penalty_px if penalty_px is not None else roi_diag[m])
                    continue
                rec["err"].append(d); rec["pen"].append(d)
                rec["adx"].append(abs(dx)); rec["ady"].append(abs(dy))
                rec["sdx"].append(dx); rec["sdy"].append(dy)
                rec["active"] += 1

        for m, rec in per_marker.items():
            if not rec["err"]:
                continue
            rows.append(dict(
                sequence=sequence, camera=camera,
                viewpoint=document.get("_meta", {}).get("view") or VIEWS.get(camera, camera),
                subject=subject_of(sequence, SUBJECT_CODES), tracker=tracker, marker=m,
                frames=len(keys), scoreable=rec["scoreable"], active=rec["active"],
                exposure=rec["active"] / rec["scoreable"] if rec["scoreable"] else 0.0,
                mean_err=statistics.mean(rec["err"]), max_err=max(rec["err"]),
                mean_adx=statistics.mean(rec["adx"]), max_adx=max(rec["adx"]),
                mean_ady=statistics.mean(rec["ady"]), max_ady=max(rec["ady"]),
                bias_dx=statistics.mean(rec["sdx"]), bias_dy=statistics.mean(rec["sdy"]),
                penalised=statistics.mean(rec["pen"]) if rec["pen"] else float("nan"),
                mis_rate=rec["mis"] / rec["reported_valid"] if rec["reported_valid"] else 0.0,
                retention_final=1.0 if rec["last_valid"] == len(keys) - 1 else 0.0,
                retention_area=rec["valid_frames"] / len(keys)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glob", required=True, help="Glob of results JSON files.")
    parser.add_argument("--out-prefix", default="subjects")
    parser.add_argument("--max-jump", type=float, default=25.0)
    parser.add_argument("--penalty-px", type=float, default=None)
    parser.add_argument("--lost-beyond", type=float, default=LOST_BEYOND_DIAMETERS,
                        help="A tracker more than this many marker diameters from its marker has "
                             "lost it (default %(default)s; 0 switches the condition off).")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched {args.glob}")
    print(f"aggregating {len(paths)} result files")

    rows = []
    for p in paths:
        rows.extend(analyse_file(p, args.max_jump, args.penalty_px, args.lost_beyond))
    if not rows:
        raise SystemExit("No trajectories produced.")

    with open(f"{args.out_prefix}_trajectories.csv", "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} trajectories -> {args.out_prefix}_trajectories.csv")

    # trajectory -> subject -> across subjects
    by_tracker = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_tracker[r["tracker"]][r["subject"]].append(r)

    summary = {}
    for tracker, subjects in by_tracker.items():
        per_subject = {}
        for s, trajs in subjects.items():
            per_subject[s] = dict(
                err=statistics.mean(t["mean_err"] for t in trajs),
                adx=statistics.mean(t["mean_adx"] for t in trajs),
                ady=statistics.mean(t["mean_ady"] for t in trajs),
                bdx=statistics.mean(t["bias_dx"] for t in trajs),
                bdy=statistics.mean(t["bias_dy"] for t in trajs),
                pen=statistics.mean(t["penalised"] for t in trajs),
                exp=statistics.mean(t["exposure"] for t in trajs),
                mis=statistics.mean(t["mis_rate"] for t in trajs),
                rfin=statistics.mean(t["retention_final"] for t in trajs),
                rarea=statistics.mean(t["retention_area"] for t in trajs))
        vals = lambda k: [v[k] for v in per_subject.values()]
        lo, hi = bootstrap_ci(vals("err"))
        summary[tracker] = dict(
            n_subjects=len(per_subject), per_subject=per_subject,
            err=statistics.mean(vals("err")), err_lo=lo, err_hi=hi,
            err_sd=statistics.pstdev(vals("err")) if len(per_subject) > 1 else 0.0,
            max_err=max(t["max_err"] for t in rows if t["tracker"] == tracker),
            adx=statistics.mean(vals("adx")), ady=statistics.mean(vals("ady")),
            max_adx=max(t["max_adx"] for t in rows if t["tracker"] == tracker),
            max_ady=max(t["max_ady"] for t in rows if t["tracker"] == tracker),
            bdx=statistics.mean(vals("bdx")), bdy=statistics.mean(vals("bdy")),
            pen=statistics.mean(vals("pen")), exp=statistics.mean(vals("exp")),
            mis=statistics.mean(vals("mis")), rfin=statistics.mean(vals("rfin")),
            rarea=statistics.mean(vals("rarea")))

    order = sorted(summary, key=lambda t: summary[t]["err"])

    # Descriptive grouping: walk the ordering, start a new group when the CI no longer overlaps
    # the group leader. Overlap of intervals is not a significance test.
    tiers, current, leader = [], [], None
    for t in order:
        if leader is None or summary[t]["err_lo"] <= summary[leader]["err_hi"]:
            current.append(t)
            if leader is None:
                leader = t
        else:
            tiers.append(current); current = [t]; leader = t
    if current:
        tiers.append(current)

    print(f"\n{'tracker':<12}{'n':>3}{'dev px':>9}{'95% CI':>18}{'mm':>7}{'expo':>7}"
          f"{'pen px':>9}{'bias dx':>9}{'bias dy':>9}{'mis':>7}")
    print("-" * 90)
    for t in order:
        s = summary[t]
        ci = "[{:.2f}, {:.2f}]".format(s["err_lo"], s["err_hi"])
        print("{:<12}{:>3}{:>9.3f}{:>18}{:>7.2f}{:>7.2f}{:>9.2f}{:>9.3f}{:>9.3f}{:>7.3f}".format(
            PRETTY.get(t, t), s["n_subjects"], s["err"], ci, s["err"] * MM_PER_PIXEL,
            s["exp"], s["pen"], s["bdx"], s["bdy"], s["mis"]))
    print("\nGroups of overlapping 95% CIs over subjects (descriptive, not a test):")
    for i, tier in enumerate(tiers, 1):
        print(f"  Group {i}: " + ", ".join(PRETTY.get(t, t) for t in tier))

    with open(f"{args.out_prefix}_summary.json", "w", encoding="utf-8") as h:
        json.dump({"summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_subject"}
                               for k, v in summary.items()},
                   "groups": [[PRETTY.get(t, t) for t in tier] for tier in tiers]}, h, indent=2)
    print(f"\nwrote {args.out_prefix}_summary.json")


if __name__ == "__main__":
    main()
