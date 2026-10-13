"""Paired, subject-level comparison of the trackers' deviations.

  python3 subject_statistics.py <prefix>_trajectories.csv [--leading ViT,DaSiamRPN,CSRT,KCF]

Reads the per-trajectory CSV of aggregate_subjects.py. The unit is the subject: each subject's value
for a tracker is the mean over that subject's trajectories, so frames and trajectories nested in a
subject are not counted as independent.

  - Omnibus: Friedman test across the trackers that every subject has, with Kendall's W for the
    agreement of the subjects' rankings (0 none, 1 complete).
  - Pairs: for each pair in a family declared in advance (every pair within the leading group, and
    every pair adjacent in the order of mean deviation), the per-subject paired differences, their
    95 % percentile bootstrap interval (10,000 resamples of subjects), the number of subjects in which
    the first tracker deviated less, and the exact two-sided sign-flip permutation p-value over all
    2^n sign patterns, Holm-corrected over the family. With n subjects the smallest attainable
    p-value is 2 / 2^n (0.016 for seven).

Standard library only.
"""

import argparse
import csv
import itertools
import math
import random
import statistics
from collections import defaultdict

from analysis_common import TRACKER_NAMES


def chi2_sf(x, k):
    """Upper tail of the chi-square distribution: regularised upper incomplete gamma Q(k/2, x/2)."""
    a, z = k / 2.0, x / 2.0
    if z <= 0:
        return 1.0
    if z < a + 1:                                       # series for P, then Q = 1 - P
        term = total = 1.0 / a
        n = a
        while abs(term) > 1e-15 * abs(total):
            n += 1
            term *= z / n
            total += term
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    b, c, d = z + 1 - a, 1e300, 1.0 / (z + 1 - a)       # continued fraction for Q
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(-z + a * math.log(z) - math.lgamma(a)) * h


def friedman(table):
    """table: subject -> tracker -> value. Returns chi2, df, p and Kendall's W."""
    trackers = sorted(set.intersection(*(set(v) for v in table.values())))
    subjects = sorted(table)
    n, k = len(subjects), len(trackers)
    rank_sums = defaultdict(float)
    for s in subjects:
        ordered = sorted(trackers, key=lambda t: table[s][t])
        i = 0
        while i < k:                                    # average ranks over ties
            j = i
            while j + 1 < k and table[s][ordered[j + 1]] == table[s][ordered[i]]:
                j += 1
            for t in ordered[i:j + 1]:
                rank_sums[t] += (i + j) / 2.0 + 1
            i = j + 1
    chi2 = 12.0 / (n * k * (k + 1)) * sum(r * r for r in rank_sums.values()) - 3 * n * (k + 1)
    return chi2, k - 1, chi2_sf(chi2, k - 1), chi2 / (n * (k - 1)), trackers, n


def sign_flip_p(diffs):
    observed = abs(statistics.mean(diffs))
    hits = sum(abs(statistics.mean(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12
               for signs in itertools.product((1, -1), repeat=len(diffs)))
    return hits / 2 ** len(diffs)


def bootstrap_ci(diffs, iterations=10000, seed=20261011):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(iterations))
    return means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectories")
    parser.add_argument("--leading", default="ViT,DaSiamRPN,CSRT,KCF")
    parser.add_argument("--metric", default="mean_err")
    parser.add_argument("--exclude", default="Hybrid (ViT+CSRT)",
                        help="comma-separated trackers left out (default: the hybrid, which is compared separately)")
    args = parser.parse_args()

    per = defaultdict(lambda: defaultdict(list))
    with open(args.trajectories, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = TRACKER_NAMES.get(row["tracker"], row["tracker"])
            if row[args.metric] not in ("", "nan") and name not in args.exclude.split(","):
                per[row["subject"]][name].append(float(row[args.metric]))
    table = {s: {t: statistics.mean(v) for t, v in trackers.items()} for s, trackers in per.items()}

    chi2, df, p, w, trackers, n = friedman(table)
    print(f"Friedman test over {n} subjects and {len(trackers)} trackers: chi2 = {chi2:.1f}, df = {df}, "
          f"p = {p:.2g}; Kendall's W = {w:.2f}")

    means = {t: statistics.mean(table[s][t] for s in table) for t in trackers}
    order = sorted(trackers, key=means.get)
    leading = [t for t in args.leading.split(",") if t in trackers]
    family = {tuple(sorted(pair, key=means.get)) for pair in itertools.combinations(leading, 2)}
    family |= {(a, b) for a, b in zip(order, order[1:])}
    family = sorted(family, key=lambda pair: (order.index(pair[0]), order.index(pair[1])))

    rows = []
    for a, b in family:
        diffs = [table[s][a] - table[s][b] for s in sorted(table)]
        lo, hi = bootstrap_ci(diffs)
        rows.append([a, b, statistics.mean(diffs), lo, hi, sum(d < 0 for d in diffs), len(diffs), sign_flip_p(diffs)])
    for row, adjusted in zip(rows, holm([r[7] for r in rows])):
        row.append(adjusted)

    print(f"\n{'pair':<26}{'mean diff':>10}{'95% bootstrap CI':>20}{'lower in':>10}{'p':>8}{'Holm p':>8}")
    for a, b, d, lo, hi, k, m, p, ph in rows:
        print(f"{a + ' - ' + b:<26}{d:>+10.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>20}{f'{k}/{m}':>10}{p:>8.3f}{ph:>8.3f}")
    print(f"\nsmallest attainable two-sided p with {n} subjects: {2 / 2 ** n:.3f}")


if __name__ == "__main__":
    main()
