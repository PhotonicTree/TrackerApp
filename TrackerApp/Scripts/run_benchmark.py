"""Runs the complete benchmark - every sequence, every camera view, every tracker - from boxes
labelled beforehand.

Workflow (run from anywhere; paths default to the TrackerApp directory above this script):

  python3 run_benchmark.py plan   --recordings <dir>   what will run, which videos still need boxes
  python3 run_benchmark.py label  --recordings <dir>   label every video without a box file, one
                                                       after another (Enter saves, Esc skips,
                                                       Q stops; see TrackerBench --label)
  python3 run_benchmark.py run    --recordings <dir>   run every tracker on every labelled video
  python3 run_benchmark.py status --recordings <dir>   progress; safe to call while a run is going

Boxes can equally be labelled in the GUI ("Label Boxes Only", or any GUI run): both write the same
box files, <labels>/<recording folder>/<video stem>.json.

Conditions (--conditions, default all):

  sparse  4-marker recordings, five camera views; all trackers, Blob Detection and PCWE references
  dense   40-marker recordings, frontal view; all trackers, Blob Detection and PCWE references
  glued   glued-marker recording, five views; Blob Detection and PCWE with the glued preset
  sweep   4-marker recordings, frontal view; boxes rebuilt at 1.5, 2.0, 2.5 and 3.0 x the
          labelled marker diameter (region-size sensitivity)

How `run` works. Every marker is tracked by its own tracker instance, so the work splits into one
TrackerBench process per (video, tracker, marker), plus one per video for the reference streams.
Up to --processes of them run at a time (default: 60 % of the logical CPUs), longest first, video
after video, so the next video starts while the last markers of the previous one finish. A tracker
process decodes the video itself and holds one tracker, a few hundred MB; GOTURN holds about
1 GB. Processes start only while their expected peak memory - measured as the run goes - fits
in the memory that was available when the run started, minus a reserve (--memory-reserve).

Each process starts from the default random state (TLD, Boosting and MIL draw random features)
and shares nothing with the others, so every result is what tracking that marker alone gives,
whatever ran next to it; TrackerBench --jobs documents the check. No latency is measured: the
processes compete for the CPU. `--frame-by-frame` runs the trackers in the frame-by-frame mode
instead, one process at a time, in tracker groups and marker chunks, as the latency
measurement does; it writes to Results/frame_by_frame unless --out is given.

`run` is resumable: every process writes its own file under <out>/.parts, and rerunning `run`
starts only what is missing. A crash or an out-of-memory kill loses one marker of one tracker.
When every process of a video is done, its outputs are merged into one results file,
<out>/<condition>/<recording folder>__<camera>.json, which the analysis scripts read.

For an unattended run that survives closing the terminal:

  setsid nohup python3 run_benchmark.py run --recordings <dir> > run.out 2>&1 &

Stopping it (pkill -f "run_benchmark.py run") also stops the processes it started.

Standard library only.
"""

import argparse
import collections
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time

from analysis_common import FRONT_CAMERA, VIEWS

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)          # model files and build-linux/ live here

CONDITIONS = ("sparse", "dense", "glued", "sweep")
EXPECTED_MARKERS = {"sparse": 4, "dense": 40, "glued": 4}
SWEEP_SCALES = (1.5, 2.0, 2.5, 3.0)

LIGHT = ["MOSSE", "KCF", "MedianFlow", "Boosting", "MIL", "CSRT", "Nano", "ViT"]
# Group 0 is always the reference streams. The groups are processes of the frame-by-frame mode.
GROUPS = {
    "sparse": [["BlobDetector", "PCWE"], LIGHT, ["TLD"], ["GOTURN"], ["DaSiamRPN"], ["Hybrid"]],
    "dense": [["BlobDetector", "PCWE"], LIGHT, ["TLD"], ["GOTURN"], ["DaSiamRPN"], ["Hybrid"]],
    "glued": [["BlobDetectorGlued", "PCWEGlued"]],
    "sweep": [["BlobDetector"], LIGHT, ["TLD"], ["GOTURN"], ["DaSiamRPN"]],
}
# Frame-by-frame mode: markers per process for trackers that load one large network per marker.
# With a 700-frame video (about 4.4 GB of decoded frames) these keep one process under about 9 GB.
CHUNK = {"GOTURN": 5, "DaSiamRPN": 10}

# Cost of one marker over one frame in ms (one thread, i9-13900H), and of the reference streams of
# one frame. They order the processes (longest first) and weight the time estimate.
COST = {"TLD": 93.0, "DaSiamRPN": 92.0, "GOTURN": 58.0, "Hybrid": 24.0, "MIL": 20.0, "CSRT": 14.0,
        "Boosting": 6.5, "Nano": 6.5, "ViT": 5.7, "MedianFlow": 4.6, "KCF": 1.5, "MOSSE": 0.1}
REFERENCE_COST = 400.0
# Peak memory of one process in GB until the run has measured it (then the largest measured peak
# plus 10 % is used). Measured on one marker of a dense video: 0.15-0.23 GB for most trackers,
# 0.39 for DaSiamRPN, 0.99 for GOTURN. The reference process holds the decoded video: about
# 4.4 GB for 700 frames.
MEMORY_GB = {"references": 4.8, "GOTURN": 1.1, "DaSiamRPN": 0.45, "TLD": 0.4}
DEFAULT_MEMORY_GB = 0.3


# --------------------------------------------------------------------------------------------
# Jobs and tasks
# --------------------------------------------------------------------------------------------

class Job:
    """One video under one condition (and, for the sweep, one box scale)."""

    def __init__(self, condition, folder, camera, video, boxes, out_dir, scale=None):
        self.condition = condition
        self.folder = folder
        self.camera = camera
        self.view = VIEWS.get(camera, camera)
        self.video = video
        self.boxes = boxes
        self.scale = scale
        self.name = f"{folder}__{camera}"
        sub = condition if scale is None else os.path.join(condition, f"s{scale:.1f}")
        self.result = os.path.join(out_dir, sub, self.name + ".json")
        self.parts_dir = os.path.join(out_dir, ".parts", sub, self.name)
        self.groups = GROUPS[condition]
        self.references = self.groups[0]
        self.trackers = [name for group in self.groups[1:] for name in group]

    def label(self):
        suffix = "" if self.scale is None else f" s={self.scale:.1f}"
        return f"{self.condition:<6} {self.folder}  {self.view}{suffix}"

    # Per-marker processes
    def references_part(self):
        return os.path.join(self.parts_dir, "references.json")

    def trackers_part(self):
        return os.path.join(self.parts_dir, "trackers.json")

    def marker_part(self, tracker, marker):
        return os.path.join(self.parts_dir, "markers", f"{tracker}_{marker:03d}.json")

    # Frame-by-frame mode
    def part(self, index):
        return os.path.join(self.parts_dir, f"group{index}.json")

    def chunks(self, index, box_total):
        """Marker ranges for one group: one range unless the group's tracker is chunked."""
        group = self.groups[index]
        size = CHUNK.get(group[0]) if len(group) == 1 else None
        if not size or box_total <= size:
            return [None]
        return [(a, min(a + size, box_total)) for a in range(0, box_total, size)]

    def chunk_part(self, index, span):
        return os.path.join(self.parts_dir, f"group{index}_markers{span[0]}-{span[1] - 1}.json")


class Task:
    """One TrackerBench process of `run`: the reference streams of a video, or one tracker on one
    marker of it."""

    def __init__(self, job, tracker=None, marker=None):
        self.job = job
        self.tracker = tracker
        self.marker = marker
        self.kind = "references" if tracker is None else tracker
        self.output = job.references_part() if tracker is None else job.marker_part(tracker, marker)
        self.cost = REFERENCE_COST if tracker is None else COST.get(tracker, 10.0)

    def describe(self):
        return "references" if self.tracker is None else f"{self.tracker} marker {self.marker}"

    def command(self, args):
        job = self.job
        names = job.references if self.tracker is None else [self.tracker]
        # The reference streams split the frames over threads; a tracker process is one job.
        command = [args.bench, "--video", job.video, "--rois", job.boxes, "--trackers", ",".join(names),
                   "--out", self.output + ".tmp", "--jobs", str(args.processes if self.tracker is None else 1)]
        if self.tracker is not None:
            command += ["--targets", f"{self.marker}:{self.marker + 1}"]
        if job.scale is not None:
            command += ["--roi-scale", str(job.scale)]
        if args.max_frames:
            command += ["--max-frames", str(args.max_frames)]
        return command


def pending_tasks(job, args):
    """The processes a video still needs: references first, then trackers longest first. With
    --redo, only the named trackers, on every marker; their results replace the earlier ones."""
    if args.redo:
        box_total = box_count(job.boxes)
        return [Task(job, tracker, marker) for tracker in args.redo if tracker in job.trackers
                for marker in range(box_total) if not os.path.exists(job.marker_part(tracker, marker))]
    tasks = []
    if args.force or args.references_only or not os.path.exists(job.references_part()):
        tasks.append(Task(job))
    if args.references_only or not job.trackers:
        return tasks
    if os.path.exists(job.trackers_part()) and not args.force:
        return tasks
    box_total = box_count(job.boxes)
    for tracker in sorted(job.trackers, key=lambda name: -COST.get(name, 10.0)):
        for marker in range(box_total):
            if args.force or not os.path.exists(job.marker_part(tracker, marker)):
                tasks.append(Task(job, tracker, marker))
    return tasks


def classify(folder):
    """Condition of a recording folder, from its name."""
    if "Dot" in folder:
        return "glued"
    if "4Marker" in folder:
        return "sparse"
    if "Markers" in folder:
        return "dense"
    return None


def build_jobs(args, conditions):
    jobs, missing_videos = [], []
    for folder in sorted(os.listdir(args.recordings)):
        if not os.path.isdir(os.path.join(args.recordings, folder)):
            continue
        kind = classify(folder)
        if kind is None:
            continue
        for condition in conditions:
            if condition == "sweep":
                if kind != "sparse":
                    continue
                cameras = [FRONT_CAMERA]
            elif condition != kind:
                continue
            else:
                cameras = [FRONT_CAMERA] if condition == "dense" else list(VIEWS)
            for camera in cameras:
                video = os.path.join(args.recordings, folder, f"Preview_{camera}.mp4")
                if not os.path.exists(video):
                    missing_videos.append(video)
                    continue
                boxes = os.path.join(args.labels, folder, f"Preview_{camera}.json")
                scales = SWEEP_SCALES if condition == "sweep" else (None,)
                for scale in scales:
                    jobs.append(Job(condition, folder, camera, video, boxes, args.out, scale))
    if args.only:
        pattern = re.compile(args.only)
        jobs = [j for j in jobs if pattern.search(j.name) or pattern.search(j.label())]
    return jobs, missing_videos


def label_jobs(jobs):
    """One labelling entry per video: the sweep reuses the sparse frontal boxes."""
    seen, unique = set(), []
    for job in jobs:
        if job.boxes in seen:
            continue
        seen.add(job.boxes)
        unique.append(job)
    return unique


def box_count(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return len(json.load(handle).get("_rois", {}))
    except (OSError, ValueError):
        return 0


# --------------------------------------------------------------------------------------------
# Logging, processes and merging
# --------------------------------------------------------------------------------------------

def log(args, message):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp}  {message}"
    print(line, flush=True)
    if args.dry_run:
        return
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "PROGRESS.log"), "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def available_memory_gb():
    """Memory available to new processes (Linux MemAvailable), or None where it is unknown."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


def wait_for_any(running):
    """Waits until one of the running processes ends; returns its pid, exit code and peak resident
    memory in GB. The peak comes from the operating system's accounting of the finished process
    (wait4), so it is exact and costs nothing; where wait4 does not exist (Windows) it is None."""
    if hasattr(os, "wait4"):
        while True:
            pid, status, usage = os.wait4(-1, 0)
            if pid in running:
                code = os.waitstatus_to_exitcode(status)
                running[pid]["process"].returncode = code
                return pid, code, usage.ru_maxrss / 1024 ** 2      # ru_maxrss is in KB on Linux
    while True:
        for pid, entry in running.items():
            code = entry["process"].poll()
            if code is not None:
                return pid, code, None
        time.sleep(0.5)


def run_process(command, output):
    """Runs one TrackerBench process to its end; returns its exit code and peak memory in GB."""
    process = subprocess.Popen(command, cwd=APP_DIR, stdout=output, stderr=subprocess.STDOUT)
    _, code, peak = wait_for_any({process.pid: {"process": process}})
    return code, peak


def write_json(document, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temporary = target + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    os.replace(temporary, target)


def merge_chunks(chunk_files, target):
    """Joins the per-marker (or per-chunk) results of one tracker into one file, in box order."""
    documents = []
    for path in chunk_files:
        with open(path, encoding="utf-8") as handle:
            documents.append(json.load(handle))
    merged = {key: value for key, value in documents[0].items() if key.startswith("_")}
    merged["_meta"].pop("target_range", None)
    merged["_diagnostics"] = {}
    for tracker in [k for k in documents[0] if not k.startswith("_")]:
        frames = [k for k in documents[0][tracker] if k.isdigit()]
        section = {key: ",".join(d[tracker][key] for d in documents if d[tracker].get(key))
                   for key in frames}
        parts = [d["_diagnostics"][tracker] for d in documents]
        diagnostics = dict(parts[0])
        for key, value in parts[0].items():
            if isinstance(value, dict):         # per-frame flags: valid, fallback_switches
                diagnostics[key] = {frame: [flag for p in parts for flag in p[key].get(frame, [])]
                                    for frame in value}
        diagnostics["n_targets"] = sum(p["n_targets"] for p in parts)
        if "fallback_switch_count" in parts[0]:
            diagnostics["fallback_switch_count"] = sum(p["fallback_switch_count"] for p in parts)
        if "frame_times_us" in parts[0]:        # frame-by-frame mode: latency of all markers
            times = [sum(values) for values in zip(*(p["frame_times_us"] for p in parts))]
            fps = len(times) / (sum(times) / 1e6) if times and sum(times) > 0 else 0.0
            diagnostics.update(frame_times_us=times, fps=fps)
            section["FPS"] = str(fps)
        diagnostics["chunks"] = len(parts)
        merged[tracker] = section
        merged["_diagnostics"][tracker] = diagnostics
    write_json(merged, target)


def merge_documents(paths, target, job=None):
    """Merges results files of the same video (different trackers) into one. With `job`, the
    result is the video's results file and records its trackers, condition and view."""
    merged = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        for key, value in document.items():
            if key == "_meta":
                meta = merged.setdefault("_meta", {})
                presets = value.pop("presets", {})
                for meta_key, meta_value in value.items():
                    meta.setdefault(meta_key, meta_value)
                meta.setdefault("presets", {}).update(presets)
            elif key == "_rois":
                if "_rois" in merged and merged["_rois"] != value:
                    raise ValueError(f"boxes differ between the parts of {path}")
                merged["_rois"] = value
            elif key == "_diagnostics":
                merged.setdefault("_diagnostics", {}).update(value)
            else:
                merged[key] = value
    if job is not None:
        merged["_meta"]["trackers"] = [k for k in merged if not k.startswith("_")]
        merged["_meta"]["condition"] = job.condition
        merged["_meta"]["view"] = job.view
    write_json(merged, target)


def finish_video(job):
    """Merges a video whose processes are all done; returns True when its results file exists.

    The per-marker files become one trackers.json next to references.json and are then removed;
    `run --references-only` merges new references with trackers.json. Trackers re-run with
    --redo replace their sections in an existing trackers.json."""
    box_total = box_count(job.boxes)
    markers = {tracker: [job.marker_part(tracker, k) for k in range(box_total)] for tracker in job.trackers}
    complete = [t for t, files in markers.items() if all(os.path.exists(path) for path in files)]
    existing = os.path.exists(job.trackers_part())
    if complete and (existing or len(complete) == len(job.trackers)):
        per_tracker = []
        for tracker in complete:
            target = os.path.join(job.parts_dir, f"{tracker}.json")
            merge_chunks(markers[tracker], target)
            per_tracker.append(target)
        # Later files override earlier ones, so re-run trackers replace their old sections.
        merge_documents(([job.trackers_part()] if existing else []) + per_tracker, job.trackers_part())
        for path in per_tracker + [path for t in complete for path in markers[t]]:
            os.remove(path)
    parts = [job.references_part()] + ([job.trackers_part()] if job.trackers else [])
    if not all(os.path.exists(path) for path in parts):
        return False
    merge_documents(parts, job.result, job)
    return True


# --------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------

def command_plan(args, jobs, missing_videos):
    unique = label_jobs(jobs)
    labelled = [j for j in unique if os.path.exists(j.boxes)]
    print(f"{len(jobs)} jobs over {len(unique)} videos; {len(labelled)} videos labelled, "
          f"{len(unique) - len(labelled)} still need boxes.\n")
    for condition in CONDITIONS:
        subset = [j for j in jobs if j.condition == condition]
        if not subset:
            continue
        done = sum(os.path.exists(j.result) for j in subset)
        processes = sum(1 + len(j.trackers) * box_count(j.boxes) for j in subset)
        print(f"  {condition:<7} {len(subset):>3} jobs, {processes:>5} TrackerBench processes, {done} done")
    print()
    for job in unique:
        count = box_count(job.boxes)
        expected = EXPECTED_MARKERS.get(job.condition, 0)
        state = "MISSING" if not os.path.exists(job.boxes) else f"{count} boxes"
        flag = "" if state == "MISSING" or count == expected else f"  (expected {expected})"
        print(f"  {state:<10} {job.label()}{flag}")
    for video in missing_videos:
        print(f"  no video   {video}")
    print("\nOblique views may legitimately show fewer than 4 markers: label only the visible ones.")


def command_label(args, jobs):
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY") and os.name != "nt":
        raise SystemExit("Labelling opens a window; no display is available.")
    todo = [j for j in label_jobs(jobs) if args.relabel or not os.path.exists(j.boxes)]
    print(f"{len(todo)} video(s) to label. Enter saves, Esc skips, Q stops; h shows help.")
    for number, job in enumerate(todo, 1):
        print(f"[{number}/{len(todo)}] {job.label()}", flush=True)
        command = [args.bench, "--label", "--video", job.video, "--save-rois", job.boxes,
                   "--expect", str(EXPECTED_MARKERS.get(job.condition, 0))]
        code = subprocess.call(command, cwd=APP_DIR)
        if code == 5:
            print("Stopped at the operator's request.")
            break
        if code not in (0, 4):
            print(f"  TrackerBench exited with code {code}")


def command_run(args, jobs):
    """Per-marker processes on all CPUs within the memory budget (see the module description)."""
    runnable = [j for j in jobs if os.path.exists(j.boxes)]
    unlabelled = len(jobs) - len(runnable)
    redo = args.force or args.references_only or bool(args.redo)
    videos = [j for j in runnable if (redo or not os.path.exists(j.result))
              and (not args.redo or set(args.redo) & set(j.trackers))]
    queue = [task for job in videos for task in pending_tasks(job, args)]
    if args.dry_run:
        for task in queue:
            print(" ".join(task.command(args)))
        print(f"{len(queue)} process(es) for {len(videos)} video(s)")
        return

    available = available_memory_gb()
    budget = args.memory_gb or ((available if available is not None else 8.0 + args.memory_reserve)
                                - args.memory_reserve)
    log(args, f"RUN start: {len(videos)} video(s), {len(queue)} process(es), up to {args.processes} "
              f"at a time within {budget:.1f} GB; {unlabelled} job(s) skipped for lack of boxes")

    remaining = collections.Counter(task.job.result for task in queue)   # result paths are unique; names are not (sweep)
    failed_videos, failures = set(), 0
    total_cost = sum(task.cost for task in queue)
    done_cost, started = 0.0, time.time()
    estimates, measured = dict(MEMORY_GB), set()
    running = {}

    videos_done = 0

    def finish(job):
        nonlocal failures, videos_done
        try:
            if finish_video(job):
                elapsed = time.time() - started
                left = elapsed / done_cost * (total_cost - done_cost) if done_cost else 0.0
                videos_done += 1        # merged in this run (earlier results may exist already)
                log(args, f"VIDEO {job.label()}  complete; {videos_done}/{len(videos)} videos, "
                          f"about {left / 3600:.1f} h left")
        except (OSError, ValueError) as error:
            failures += 1
            log(args, f"FAIL  merge {job.label()}: {error}")

    def fits(task):
        if not running:
            return True
        committed = sum(estimates.get(entry["task"].kind, DEFAULT_MEMORY_GB) for entry in running.values())
        if committed + estimates.get(task.kind, DEFAULT_MEMORY_GB) > budget:
            return False
        free = available_memory_gb()
        return free is None or free >= args.memory_reserve

    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)

    for job in videos:                      # every process done already, merge missing
        if remaining[job.result] == 0:
            finish(job)
    try:
        while queue or running:
            while queue and len(running) < args.processes:
                task = next((t for t in queue if fits(t)), None)
                if task is None:
                    break
                queue.remove(task)
                os.makedirs(os.path.dirname(task.output), exist_ok=True)
                handle = open(task.output + ".log", "w", encoding="utf-8")
                process = subprocess.Popen(task.command(args), cwd=APP_DIR, stdout=handle,
                                           stderr=subprocess.STDOUT)
                running[process.pid] = {"task": task, "process": process, "log": handle, "start": time.time()}
            pid, code, peak = wait_for_any(running)
            entry = running.pop(pid)
            entry["log"].close()
            task, job = entry["task"], entry["task"].job
            seconds = time.time() - entry["start"]
            memory = "" if peak is None else f"  peak {peak:4.2f} GB"
            if peak is not None:
                estimate = peak * 1.1
                estimates[task.kind] = estimate if task.kind not in measured else max(estimates[task.kind], estimate)
                measured.add(task.kind)
            if code == 0 and os.path.exists(task.output + ".tmp"):
                os.replace(task.output + ".tmp", task.output)
                os.remove(task.output + ".log")
                log(args, f"done  {job.label()}  [{task.describe()}]  {seconds:7.1f} s{memory}")
            else:
                failures += 1
                failed_videos.add(job.result)
                log(args, f"FAIL  {job.label()}  [{task.describe()}]  exit code {code}{memory}; "
                          f"see {task.output}.log")
            done_cost += task.cost
            remaining[job.result] -= 1
            if remaining[job.result] == 0 and job.result not in failed_videos:
                finish(job)
    except KeyboardInterrupt:
        for entry in running.values():
            entry["process"].terminate()
        for entry in running.values():
            entry["process"].wait()
            entry["log"].close()
        log(args, f"RUN stopped; {len(running)} running process(es) ended, rerun `run` to continue.")
        return
    log(args, f"RUN end: {failures} failure(s); rerun `run` to retry them.")


def command_run_frame_by_frame(args, jobs):
    """One process at a time, in tracker groups and marker chunks, frame by frame (latency)."""
    runnable = [j for j in jobs if os.path.exists(j.boxes)]
    unlabelled = len(jobs) - len(runnable)
    # --references-only recomputes the reference streams (group 0) of every labelled video and
    # merges them with the tracker outputs already written; the trackers are not run again.
    redo = args.force or args.references_only
    selected = lambda job: [0] if args.references_only else range(len(job.groups))
    pending = [j for j in runnable if redo or not os.path.exists(j.result)]
    total_groups = sum(1 for j in pending for i in selected(j) if redo or not os.path.exists(j.part(i)))
    log(args, f"RUN start (frame by frame): {len(pending)} job(s), {total_groups} group(s); "
              f"{unlabelled} job(s) skipped for lack of boxes; threads={args.threads}")

    started, finished_groups, failures = time.time(), 0, 0
    for job in pending:
        if not args.dry_run:
            os.makedirs(job.parts_dir, exist_ok=True)
        job_failed = False
        box_total = box_count(job.boxes)
        for index, group in enumerate(job.groups):
            if index not in selected(job):
                continue
            part = job.part(index)
            if os.path.exists(part) and not redo:
                continue
            spans = job.chunks(index, box_total)
            chunk_files = []
            for span in spans:
                output_file = part if span is None else job.chunk_part(index, span)
                chunk_files.append(output_file)
                if os.path.exists(output_file) and not redo:
                    continue
                command = [args.bench, "--video", job.video, "--rois", job.boxes,
                           "--trackers", ",".join(group), "--out", output_file + ".tmp",
                           "--threads", str(args.threads)]
                if span is not None:
                    command += ["--targets", f"{span[0]}:{span[1]}"]
                if job.scale is not None:
                    command += ["--roi-scale", str(job.scale)]
                if args.max_frames:
                    command += ["--max-frames", str(args.max_frames)]
                if args.dry_run:
                    print(" ".join(command))
                    continue
                what = ",".join(group) + ("" if span is None else f" markers {span[0]}-{span[1] - 1}")
                group_start = time.time()
                with open(output_file + ".log", "w", encoding="utf-8") as output:
                    code, peak = run_process(command, output)
                seconds = time.time() - group_start
                memory = "" if peak is None else f"  peak {peak:4.1f} GB"
                if code == 0 and os.path.exists(output_file + ".tmp"):
                    os.replace(output_file + ".tmp", output_file)
                    log(args, f"done  {job.label()}  [{what}]  {seconds:7.1f} s{memory}")
                else:
                    job_failed = True
                    failures += 1
                    log(args, f"FAIL  {job.label()}  [{what}]  exit code {code}{memory}; see {output_file}.log")
            if args.dry_run:
                continue
            finished_groups += 1
            if spans != [None] and all(os.path.exists(f) for f in chunk_files):
                merge_chunks(chunk_files, part)
            if os.path.exists(part):
                elapsed = time.time() - started
                remaining = elapsed / finished_groups * (total_groups - finished_groups)
                log(args, f"      {job.label()}  group {index + 1}/{len(job.groups)} complete; "
                          f"{finished_groups}/{total_groups} groups, about {remaining / 3600:.1f} h left")
        complete = all(os.path.exists(job.part(i)) for i in range(len(job.groups)))
        if not job_failed and not args.dry_run and complete:
            try:
                merge_documents([job.part(i) for i in range(len(job.groups))], job.result, job)
            except (OSError, ValueError) as error:
                log(args, f"FAIL  merge {job.label()}: {error}")
                failures += 1
    log(args, f"RUN end: {failures} failure(s); rerun to retry failed groups.")


def command_status(args, jobs):
    unique = label_jobs(jobs)
    print(f"videos labelled: {sum(os.path.exists(j.boxes) for j in unique)}/{len(unique)}")
    for condition in CONDITIONS:
        subset = [j for j in jobs if j.condition == condition]
        if not subset:
            continue
        done = sum(os.path.exists(j.result) for j in subset)
        total = processes_done = 0
        for job in subset:
            count = 1 + len(job.trackers) * box_count(job.boxes)
            total += count
            if os.path.exists(job.result):
                processes_done += count
                continue
            processes_done += os.path.exists(job.references_part())
            if os.path.exists(job.trackers_part()):
                processes_done += count - 1
            elif os.path.isdir(os.path.join(job.parts_dir, "markers")):
                processes_done += sum(name.endswith(".json") for name in
                                      os.listdir(os.path.join(job.parts_dir, "markers")))
        print(f"  {condition:<7} videos {done:>3}/{len(subset):<3}  processes {processes_done:>5}/{total}")
    progress = os.path.join(args.out, "PROGRESS.log")
    if os.path.exists(progress):
        with open(progress, encoding="utf-8") as handle:
            lines = handle.readlines()
        failed = [line for line in lines if "  FAIL  " in line]
        print(f"\nlast entries of {progress}:")
        for line in lines[-5:]:
            print("  " + line.rstrip())
        if failed:
            print(f"\n{len(failed)} failure(s) logged; rerun `run` to retry them.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "label", "run", "status"))
    parser.add_argument("--recordings", default=os.path.join(APP_DIR, "Recordings"),
                        help="Directory holding one folder per recording with Preview_<camera>.mp4 files.")
    parser.add_argument("--labels", default=os.path.join(APP_DIR, "Labels"),
                        help="Root of the box files (default: Labels/ next to the models, as the GUI uses).")
    parser.add_argument("--out", help="Results directory (default: Results/benchmark, or "
                                      "Results/frame_by_frame with --frame-by-frame).")
    parser.add_argument("--bench", default=os.path.join(APP_DIR, "build-linux", "TrackerBench"))
    parser.add_argument("--conditions", default=",".join(CONDITIONS),
                        help="Comma-separated subset of: " + ", ".join(CONDITIONS))
    parser.add_argument("--only", help="Regular expression; only jobs whose name matches.")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) * 3 // 5),
                        help="TrackerBench processes at a time (default: 60 %% of the logical CPUs, "
                             "12 of 20). On an i9-13900H under WSL the throughput is the same from "
                             "8 to 16 processes and half as high with all 20.")
    parser.add_argument("--memory-reserve", type=float, default=1.5,
                        help="GB kept free for the rest of the system (default 1.5).")
    parser.add_argument("--memory-gb", type=float,
                        help="Memory budget for the processes in GB (default: available memory "
                             "at the start minus the reserve).")
    parser.add_argument("--frame-by-frame", action="store_true",
                        help="run: frame-by-frame processes one at a time, as the latency "
                             "measurement runs them.")
    parser.add_argument("--threads", type=int, default=-1,
                        help="OpenCV threads per process with --frame-by-frame (default -1, the "
                             "OpenCV default); latency is measured with 1.")
    parser.add_argument("--max-frames", type=int, default=0, help="Truncate sequences (test runs).")
    parser.add_argument("--relabel", action="store_true", help="label: also videos that have boxes.")
    parser.add_argument("--force", action="store_true", help="run: redo processes that already ran.")
    parser.add_argument("--redo", help="run: re-run only these trackers (comma-separated, e.g. Hybrid) "
                                       "on every marker and replace their results.")
    parser.add_argument("--references-only", action="store_true",
                        help="run: recompute only the reference streams and re-merge with existing tracker outputs.")
    parser.add_argument("--dry-run", action="store_true", help="run: print the commands only.")
    args = parser.parse_args()

    args.redo = [t for t in (args.redo or "").split(",") if t]
    if args.out is None:
        args.out = os.path.join(APP_DIR, "Results", "frame_by_frame" if args.frame_by_frame else "benchmark")
    args.recordings = os.path.abspath(args.recordings)
    args.labels = os.path.abspath(args.labels)
    args.out = os.path.abspath(args.out)
    args.bench = os.path.abspath(args.bench)
    if not os.path.isdir(args.recordings):
        raise SystemExit(f"Recordings directory not found: {args.recordings} (pass --recordings)")
    conditions = [c for c in args.conditions.split(",") if c]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise SystemExit(f"Unknown condition(s): {', '.join(sorted(unknown))}")

    jobs, missing_videos = build_jobs(args, conditions)
    if args.command == "plan":
        command_plan(args, jobs, missing_videos)
    elif args.command == "label":
        command_label(args, jobs)
    elif args.command == "run":
        if not os.path.exists(args.bench):
            raise SystemExit(f"TrackerBench not found: {args.bench} (build it first, see README.md)")
        if args.frame_by_frame:
            command_run_frame_by_frame(args, jobs)
        else:
            command_run(args, jobs)
    else:
        command_status(args, jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
