"""Compares the hybrid pipeline against its components on the multi-target retention condition.

Reports, per tracker and per subject, the metrics that matter for this comparison: how many marker
identities survive to the end, how much of the sequence they survive, how often a tracker sits
closer to a neighbouring marker than to its own, and the per-frame deviation. Aggregates to the
subject level, since subjects are the independent unit.

A marker is lost, for good, once its tracker has failed for LOSS_FRAMES consecutive frames (10, a
sixth of a second at 60 fps): it reports failure, sits nearer another marker than its own, or
drifts more than two marker diameters off it (the failure conditions of
analysis_common.tracking_state). Shorter failures are not counted as losses, so a single frame in
which the reference itself is wrong cannot end a marker. The deviation is the mean over active
frames.

Standard library only.
"""

import argparse
import glob
import statistics
from collections import defaultdict

from analysis_common import (LOST_BEYOND_DIAMETERS, REFERENCE_SECTIONS, TRACKER_NAMES,
                             build_reference_tracks, frame_keys, load_results, load_subject_codes,
                             marker_diameters, parse_points, roi_centres, subject_of, tracking_state)

PRETTY = TRACKER_NAMES
SUBJECT_CODES = load_subject_codes()
METRICS = ("retention_final", "retention_area", "mis_rate", "mean_dev")
LOSS_FRAMES = 10


def subject_means(per_sequence):
    """Averages each metric over one subject's sequences, so that every subject counts once."""
    out = {}
    for key in METRICS:
        values = [m[key] for m in per_sequence if m[key] == m[key]]
        out[key] = statistics.mean(values) if values else float("nan")
    return out


def per_file(path, max_jump=25.0, lost_beyond=LOST_BEYOND_DIAMETERS, loss_frames=LOSS_FRAMES):
    doc = load_results(path)
    seeds = roi_centres(doc)
    diameters = marker_diameters(doc)
    blob = doc.get("MultiBlobDetector")
    if not seeds or blob is None:
        return {}
    det = {k: parse_points(blob[k]) for k in frame_keys(blob)}
    ref = build_reference_tracks(det, seeds, max_jump_px=max_jump)

    out = {}
    for tr in [t for t in doc if not t.startswith("_") and t not in REFERENCE_SECTIONS]:
        sec = doc[tr]
        keys = frame_keys(sec)
        if not keys:
            continue
        valid = doc.get("_diagnostics", {}).get(tr, {}).get("valid", {})
        n = len(seeds)
        alive = [True] * n          # still correctly tracking its own marker
        failing = [0] * n           # consecutive failed frames so far
        survived = [0] * n          # frames, among those with a reference, still correct
        referenced = [0] * n        # frames in which the reference sees the marker
        errs = []
        mis_total = reported_valid_total = 0

        for idx, k in enumerate(keys):
            tg = parse_points(sec[k])
            fl = valid.get(k, [True] * len(tg))
            rf = ref.get(k, [None] * n)
            for m in range(min(len(tg), n)):
                ok = bool(fl[m]) if m < len(fl) else True
                state, d, _, _, _ = tracking_state(tg[m], m, ok, rf, diameters[m], lost_beyond)
                if state == "unscored":
                    continue
                referenced[m] += 1
                reported_valid_total += state != "reported"
                mis_total += state == "swap"
                if state != "active":
                    failing[m] += 1
                    if failing[m] >= loss_frames:
                        alive[m] = False  # a failure that lasts is permanent for retention
                    continue
                failing[m] = 0
                errs.append(d)
                if alive[m]:
                    survived[m] += 1
        # A marker the reference never sees cannot be scored; frames without a reference are
        # left out of that marker's retention area rather than counted as lost.
        scored = [m for m in range(n) if referenced[m] > 0]
        if not scored:
            continue
        out[tr] = dict(
            retention_final=sum(1 for m in scored if alive[m]) / len(scored),
            retention_area=sum(survived[m] / referenced[m] for m in scored) / len(scored),
            mis_rate=mis_total / reported_valid_total if reported_valid_total else 0.0,
            mean_dev=statistics.mean(errs) if errs else float("nan"),
            n_markers=len(scored), n_unscored=n - len(scored))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("glob", nargs="?", default="../Results/benchmark/dense/*.json",
                        help="results files (default: the 40-marker benchmark results)")
    parser.add_argument("--lost-beyond", type=float, default=LOST_BEYOND_DIAMETERS,
                        help="A tracker more than this many marker diameters from its marker has "
                             "lost it (default %(default)s; 0 switches the condition off).")
    parser.add_argument("--loss-frames", type=int, default=LOSS_FRAMES,
                        help="Consecutive failed frames after which a marker counts as lost for good "
                             "(default %(default)s; 1 = the first failed frame).")
    args = parser.parse_args()
    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit("no result files")
    per_sequence = defaultdict(lambda: defaultdict(list))
    for p in paths:
        subj = subject_of(p, SUBJECT_CODES)
        for tr, m in per_file(p, lost_beyond=args.lost_beyond, loss_frames=args.loss_frames).items():
            per_sequence[tr][subj].append(m)
    by_tracker = {tr: {s: subject_means(ms) for s, ms in subjects.items()}
                  for tr, subjects in per_sequence.items()}
    print(f"{len(paths)} sequences, subjects: "
          f"{sorted({subject_of(p, SUBJECT_CODES) for p in paths})}\n")

    hdr = f"{'tracker':<18}{'n':>3}{'retain_final':>14}{'retain_area':>13}{'misassign':>11}{'dev_px':>9}"
    print(hdr); print("-" * len(hdr))
    rows = {}
    for tr, per in by_tracker.items():
        rf = [v["retention_final"] for v in per.values()]
        ra = [v["retention_area"] for v in per.values()]
        ms = [v["mis_rate"] for v in per.values()]
        dv = [v["mean_dev"] for v in per.values() if v["mean_dev"] == v["mean_dev"]]
        # n is the number of subjects; each subject's sequences were averaged first.
        rows[tr] = (statistics.mean(rf), statistics.mean(ra), statistics.mean(ms),
                    statistics.mean(dv) if dv else float("nan"), len(per), rf)
    for tr in sorted(rows, key=lambda t: -rows[t][0]):
        f, a, m, d, n, _ = rows[tr]
        print(f"{PRETTY.get(tr,tr):<18}{n:>3}{f*100:>13.1f}%{a*100:>12.1f}%{m*100:>10.2f}%{d:>9.3f}")

    if "HybridVitCsrt" in rows:
        print("\nper-subject final retention (%), hybrid vs components:")
        subs = sorted(by_tracker["HybridVitCsrt"])
        print(f"  {'subject':<10}" + "".join(f"{PRETTY.get(t,t):>18}" for t in
              ["TrackerVit", "TrackerCSRT", "TrackerDaSiamRPN", "HybridVitCsrt"] if t in by_tracker))
        for s in subs:
            line = f"  {s:<10}"
            for t in ["TrackerVit", "TrackerCSRT", "TrackerDaSiamRPN", "HybridVitCsrt"]:
                if t in by_tracker:
                    v = by_tracker[t].get(s)
                    line += f"{(v['retention_final']*100 if v else float('nan')):>18.1f}"
            print(line)


if __name__ == "__main__":
    main()
