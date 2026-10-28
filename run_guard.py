"""Real-time safety analytics: pose + tracking + (optional) segmentation -> events (intrusion, loitering, crowding, fall).

Usage:
    python run_guard.py --source video.mp4 --zone "0.1,0.4;0.9,0.4;0.9,0.95;0.1,0.95" --seg models/yolov8n-seg.onnx
"""
import argparse
import json
import time

import cv2
import numpy as np

from visionguard import EventConfig, EventEngine, PoseDetector, SegDetector, SORTTracker
from visionguard.models import SKELETON


def parse_zone(s, W, H):
    if not s:
        return None
    pts = [tuple(map(float, p.split(","))) for p in s.split(";")]
    return np.array([[x * W, y * H] for x, y in pts], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="models/yolov8n-pose.onnx")
    ap.add_argument("--seg", default=None, help="optional segmentation model for mask overlay + zone occupancy")
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", default="guard_out.mp4")
    ap.add_argument("--events", default="events.jsonl")
    ap.add_argument("--zone", default=None, help='normalized polygon "x,y;x,y;..." (omit = whole frame)')
    ap.add_argument("--loiter-s", type=float, default=5.0)
    ap.add_argument("--crowd-n", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.35)
    a = ap.parse_args()

    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    zone = parse_zone(a.zone, W, H)

    pose = PoseDetector(a.pose, conf=a.conf)
    seg = SegDetector(a.seg, conf=a.conf) if a.seg else None
    trk = SORTTracker()
    eng = EventEngine(fps, zone, EventConfig(loiter_s=a.loiter_s, crowd_n=a.crowd_n))
    all_events, n, t0, banner = [], 0, time.perf_counter(), ("", 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, kpts = pose(frame)
        tracks = trk.update(boxes[:, :4], boxes[:, 4], list(kpts) if len(kpts) else [])
        events = eng.update(n, tracks)
        all_events += events
        if events:
            banner = (", ".join(sorted({e["type"].upper() for e in events})), n + int(fps))

        if seg is not None:
            _, masks = seg(frame)
            for m in masks:
                frame[m] = (0.6 * frame[m] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
        if zone is not None:
            cv2.polylines(frame, [zone.astype(np.int32)], True, (0, 255, 255), 2)
        for tid, b, sc, k in tracks:
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 160, 0), 2)
            cv2.putText(frame, f"#{tid}", (int(b[0]), max(14, int(b[1]) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 0), 2)
            if k is not None:
                for i, j in SKELETON:
                    if k[i, 2] > 0.3 and k[j, 2] > 0.3:
                        cv2.line(frame, tuple(k[i, :2].astype(int)), tuple(k[j, :2].astype(int)), (0, 255, 0), 2)
        if n < banner[1]:
            cv2.putText(frame, "EVENT: " + banner[0], (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        out.write(frame)
        n += 1
    cap.release(); out.release()
    with open(a.events, "w") as f:
        for e in all_events:
            f.write(json.dumps(e) + "\n")
    dt = time.perf_counter() - t0
    print(f"frames={n} fps={n/dt:.1f} events={len(all_events)} pose_infer_mean={np.mean(pose.infer_ms):.1f} ms"
          + (f" seg_infer_mean={np.mean(seg.infer_ms):.1f} ms" if seg else ""))
    for e in all_events:
        print(" ", e)


if __name__ == "__main__":
    main()
