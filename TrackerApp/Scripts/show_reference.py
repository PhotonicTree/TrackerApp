"""Shows the Blob Detection reference of a results file, for visual checking.

  python show_reference.py --results <results.json> [--show | --snapshot <frame> | --out <file.mp4>]
                           [--video <recording>] [--trackers CSRT,KCF] [--pcwe] [--full-frame]

Left panel: the reference alone on a white background. A filled coloured dot with its number is the
reference position of that marker, with a short trail of its previous positions; small grey dots are
Blob Detection detections not assigned to any marker; a red "n?" marks marker n when it has no
reference in that frame, drawn at its last known position. Right panel: the same on the recording,
with every raw detection as a small white cross. The face region the detector searched is outlined.

The reference is built exactly as the analysis builds it (analysis_common.build_reference_tracks:
seeded from the labelled marker centres, each marker following its own previous position), so what
is shown is what the deviations are measured against.

Modes (default: write a video next to the results file, *_reference.mp4):
  --show          interactive window: space play/pause, d or right arrow next frame, a or left arrow
                  previous frame, s save the current frame as PNG, q or Esc quit
  --snapshot n    write a PNG of frame n
  --out FILE      output path for the video or snapshot

Options:
  --video         the recording (default: the path stored in the results file)
  --trackers      also draw these trackers' boxes on the recording, e.g. CSRT,ViT
  --pcwe          also draw the PCWE detections (blue)
  --full-frame    show the whole frame instead of the searched face region
  --start/--end   frame range of the video; --height panel height in pixels (default 900)

Needs OpenCV for Python (pip install opencv-python); everything else is standard library.
"""

import argparse
import colorsys
import os
import sys

import cv2
import numpy as np

from analysis_common import build_reference_tracks, frame_keys, load_results, parse_points, roi_centres

MISSING = (0, 0, 220)
TRACKER_COLOURS = [((0, 0, 0), "black"), ((200, 0, 200), "magenta"), ((0, 140, 255), "orange"),
                   ((140, 140, 0), "teal"), ((0, 0, 140), "dark red")]


def palette(n):
    """Distinct marker colours, none of them red: red is reserved for a marker without reference."""
    colours = []
    for i in range(max(n, 1)):
        hue = 0.08 + (i * 0.618034) % 0.84
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.8)
        colours.append((int(b * 255), int(g * 255), int(r * 255)))
    return colours


def label(img, s, org, colour, background, scale=0.45):
    """Text on a filled background box, readable on both the white and the video panel."""
    (w, h), baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 1, y - h - 2), (x + w + 1, y + baseline), background, -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


class Reference:
    """The reference and everything drawn with it, for one results file."""

    def __init__(self, document, max_jump, with_pcwe, tracker_names):
        if "MultiBlobDetector" not in document:
            raise SystemExit("The results file has no Blob Detection stream (MultiBlobDetector).")
        self.document = document
        self.seeds = roi_centres(document)
        self.rois = [document["_rois"][k] for k in sorted(document["_rois"], key=int)]
        blobs = document["MultiBlobDetector"]
        self.detections = {k: parse_points(blobs[k]) for k in frame_keys(blobs)}
        self.tracks = build_reference_tracks(self.detections, self.seeds, max_jump_px=max_jump)
        edges = document.get("WeightedEdgesExtractor", {})
        self.pcwe = {k: parse_points(edges[k]) for k in frame_keys(edges)} if with_pcwe else {}
        self.trackers = []
        for name in tracker_names:
            section = next((k for k in document if k.lower() in (name.lower(), "tracker" + name.lower())), None)
            if section is None:
                print(f"warning: the results file has no tracker {name}", file=sys.stderr)
            else:
                self.trackers.append(section)
        self.colours = palette(len(self.seeds))

        # Last known reference position of every marker up to each frame, for the "n?" markers.
        self.last_known, current = {}, list(self.seeds)
        for key in sorted(self.tracks):
            current = [p if p else q for p, q in zip(self.tracks[key], current)]
            self.last_known[key] = current

    def at(self, i):
        return self.tracks.get(f"{i:04d}", [None] * len(self.seeds))

    def trail(self, m, i, length):
        return [p[m] for p in (self.at(j) for j in range(max(0, i - length), i + 1)) if p[m]]


def view_window(document, frame_size, full_frame, pad=30):
    """The part of the frame shown: the searched face region with a margin, or the whole frame."""
    width, height = frame_size
    region = document.get("_meta", {}).get("reference_detector", {}).get("MultiBlobDetector", {}).get("region")
    if full_frame or not region:
        return 0, 0, width, height
    x, y, w, h = region
    return max(0, x - pad), max(0, y - pad), min(width, x + w + pad), min(height, y + h + pad)


def render(ref, i, frame, window, height, trail_length, region):
    x0, y0, x1, y1 = window
    scale = height / (y1 - y0)
    width = int(round((x1 - x0) * scale))
    right = cv2.resize(frame[y0:y1, x0:x1], (width, height), interpolation=cv2.INTER_AREA)
    left = np.full_like(right, 255)

    def to(p):
        return int(round((p[0] - x0) * scale)), int(round((p[1] - y0) * scale))

    key = f"{i:04d}"
    if region:
        x, y, w, h = region
        for panel in (left, right):
            cv2.rectangle(panel, to((x, y)), to((x + w, y + h)), (170, 170, 170), 1)

    reference = ref.at(i)
    assigned = {tuple(p) for p in reference if p}
    for p in ref.detections.get(key, []):
        if tuple(p) not in assigned:
            cv2.circle(left, to(p), 3, (160, 160, 160), -1, cv2.LINE_AA)
        cv2.drawMarker(right, to(p), (255, 255, 255), cv2.MARKER_CROSS, 7, 1)
    for p in ref.pcwe.get(key, []):
        for panel in (left, right):
            cv2.circle(panel, to(p), 2, (255, 110, 0), -1, cv2.LINE_AA)
    for t, section in enumerate(ref.trackers):
        for box in parse_points(ref.document[section].get(key, "")):
            if len(box) >= 4:
                cx, cy, w, h = box[:4]
                cv2.rectangle(right, to((cx - w / 2, cy - h / 2)), to((cx + w / 2, cy + h / 2)),
                              TRACKER_COLOURS[t % len(TRACKER_COLOURS)][0], 1, cv2.LINE_AA)

    found = 0
    for m, p in enumerate(reference):
        radius = max(4, int(round(0.5 * ref.rois[m].get("d", 12) * scale)))
        if p:
            found += 1
            points = np.array([to(q) for q in ref.trail(m, i, trail_length)], np.int32)
            for panel, fill, background in ((left, -1, (255, 255, 255)), (right, 2, (30, 30, 30))):
                if len(points) > 1:
                    cv2.polylines(panel, [points], False, ref.colours[m], 1, cv2.LINE_AA)
                cv2.circle(panel, to(p), radius, ref.colours[m], fill, cv2.LINE_AA)
                label(panel, str(m), (to(p)[0] + radius + 3, to(p)[1] - radius), ref.colours[m], background)
        else:
            q = ref.last_known.get(key, ref.seeds)[m]
            for panel, background in ((left, (255, 255, 255)), (right, (30, 30, 30))):
                cv2.circle(panel, to(q), radius, MISSING, 1, cv2.LINE_AA)
                label(panel, f"{m}?", (to(q)[0] + radius + 3, to(q)[1] - radius), MISSING, background)

    bar = np.full((52, 2 * width + 6, 3), 245, np.uint8)
    status = (f"frame {i}   detections {len(ref.detections.get(key, []))}   "
              f"markers with reference {found}/{len(ref.seeds)}")
    if ref.trackers:
        status += "   boxes: " + ", ".join(f"{s} ({TRACKER_COLOURS[t % len(TRACKER_COLOURS)][1]})"
                                         for t, s in enumerate(ref.trackers))
    legend = ("left: reference only (coloured = marker n, grey = unassigned detection, red n? = no reference)   "
              "right: on the recording (white + = raw detection)" + ("   blue = PCWE" if ref.pcwe else ""))
    label(bar, status, (8, 20), (30, 30, 30), (245, 245, 245), 0.5)
    label(bar, legend, (8, 42), (90, 90, 90), (245, 245, 245), 0.42)
    return np.vstack([bar, np.hstack([left, np.full((height, 6, 3), 90, np.uint8), right])])


def show(ref, capture, window, args, region, frames_total, fps, out_stem):
    name = "Reference (space play/pause, a/d step, s save, q quit)"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    i, playing, position, frame, sized, seen_visible = args.start, False, None, None, False, False
    while True:
        if position != i:
            if position is None or i != position + 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = capture.read()
            if not ok:
                if i == 0:
                    raise SystemExit("The recording could not be read.")
                i, playing, position = i - 1, False, None
                continue
            position = i
            canvas = render(ref, i, frame, window, args.height, args.trail, region)
            if not sized:
                factor = min(1.0, 1700 / canvas.shape[1], 960 / canvas.shape[0])
                cv2.resizeWindow(name, int(canvas.shape[1] * factor), int(canvas.shape[0] * factor))
                sized = True
            cv2.imshow(name, canvas)
        key = cv2.waitKeyEx(max(1, int(1000 / fps)) if playing else 50)
        visible = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
        seen_visible = seen_visible or visible >= 1     # GTK reports -1 throughout: never trust that
        if seen_visible and visible < 1:
            break
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            playing = not playing
        elif key in (ord("d"), 65363, 2555904, 16777236):       # right arrow: GTK, Windows, Qt
            i, playing = min(frames_total - 1, i + 1), False
        elif key in (ord("a"), 65361, 2424832, 16777234):       # left arrow
            i, playing = max(0, i - 1), False
        elif key == ord("s"):
            path = f"{out_stem}_reference_{i:04d}.png"
            cv2.imwrite(path, canvas)
            print(f"saved {path}")
        elif playing:
            i = i + 1 if i + 1 < frames_total else i
            playing = i + 1 < frames_total
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--video")
    parser.add_argument("--out")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--snapshot", type=int)
    parser.add_argument("--trackers", default="")
    parser.add_argument("--pcwe", action="store_true")
    parser.add_argument("--full-frame", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--trail", type=int, default=20, help="frames of trail behind each marker")
    parser.add_argument("--max-jump", type=float, default=25.0)
    args = parser.parse_args()

    document = load_results(args.results)
    video = args.video or document.get("_meta", {}).get("sequence")
    if not video or not os.path.exists(video):
        raise SystemExit(f"Recording not found: {video!r}; pass --video.")
    ref = Reference(document, args.max_jump, args.pcwe, [t.strip() for t in args.trackers.split(",") if t.strip()])

    capture = cv2.VideoCapture(video)
    frames_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    window = view_window(document, frame_size, args.full_frame)
    region = document.get("_meta", {}).get("reference_detector", {}).get("MultiBlobDetector", {}).get("region")
    out_stem = os.path.splitext(args.out or args.results)[0]
    args.start = max(0, min(args.start, frames_total - 1))

    if args.show:
        show(ref, capture, window, args, region, frames_total, fps, out_stem)
        return

    if args.snapshot is not None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.snapshot)
        ok, frame = capture.read()
        if not ok:
            raise SystemExit(f"Frame {args.snapshot} could not be read.")
        path = args.out or f"{out_stem}_reference_{args.snapshot:04d}.png"
        cv2.imwrite(path, render(ref, args.snapshot, frame, window, args.height, args.trail, region))
        print(f"wrote {path}")
        return

    end = min(frames_total, args.end if args.end is not None else frames_total)
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    path = args.out or f"{out_stem}_reference.mp4"
    writer = None
    for i in range(args.start, end):
        ok, frame = capture.read()
        if not ok:
            break
        canvas = render(ref, i, frame, window, args.height, args.trail, region)
        if writer is None:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.shape[1], canvas.shape[0]))
            if not writer.isOpened():
                raise SystemExit(f"Could not open a video writer for {path}.")
        writer.write(canvas)
        if i % 100 == 0:
            print(f"frame {i}/{end - 1}", flush=True)
    if writer is not None:
        writer.release()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
