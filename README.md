# VisionGuard: Real-Time Video Safety Analytics (Pose, Segmentation, Tracking, Event Detection)

A production-style video analytics pipeline that turns camera frames into **safety events**: restricted-zone intrusion, loitering, crowding, and fall detection. It combines **YOLOv8 pose estimation**, **instance segmentation**, and a **Kalman-filter multi-object tracker**, all running on **ONNX Runtime**, with an **event-level evaluation framework** (precision / recall / F1 with temporal tolerance).

## Pipeline

```
frame -> YOLOv8-pose (ONNX) -> person boxes + 17 keypoints
      -> SORT tracker (Kalman + Hungarian), keypoints carried per track
      -> Event engine: intrusion / loitering / crowding / fall
      -> (optional) YOLOv8-seg (ONNX): person masks for overlay / occupancy
      -> annotated video + events.jsonl
```

| Component | File | Details |
|---|---|---|
| Model export | `export_models.py` | YOLOv8 pose + seg -> ONNX (opset 13) |
| Pose / segmentation inference | `visionguard/models.py` | ONNX Runtime; letterbox, output decoding, NMS, keypoint mapping, mask reconstruction from prototypes (sigmoid(coeffs . protos), un-letterbox, box crop) |
| Tracker | `visionguard/tracker.py` | 7-state Kalman filter, IoU cost matrix, Hungarian assignment; carries per-detection payload (keypoints) |
| Event engine | `visionguard/events.py` | Polygon zones (`pointPolygonTest` on foot point), dwell-time loitering, crowd threshold, pose-based fall detection |
| Evaluation | `evaluate.py` | Greedy temporal matching, per-type TP/FP/FN, precision, recall, F1 |
| Tests | `tests/test_events.py` | Torso angle, intrusion + loitering, fall transition logic, crowding edge-trigger, metrics |

## Fall detection logic

Torso angle = angle of the shoulder-midpoint -> hip-midpoint axis from vertical. A fall is raised when a **tracked** person (1) was upright (<35 deg) within the last 2 s, and (2) stays horizontal (>60 deg, or bbox aspect ratio > 1.3 when keypoints are occluded) for >= 0.5 s. Requiring the upright -> horizontal *transition* avoids alarms for people already lying down, and the hold time suppresses single-frame pose glitches.

## Quick start

```bash
pip install -r requirements.txt
python export_models.py
python run_guard.py --source video.mp4 --seg models/yolov8n-seg.onnx \
    --zone "0.1,0.4;0.9,0.4;0.9,0.95;0.1,0.95" --events events.jsonl
python evaluate.py --pred events.jsonl --gt ground_truth.json --tol 2.0
pytest -q
```

`ground_truth.json`: `[{"type": "fall", "t": 12.4}, {"type": "intrusion", "t": 3.1}]`

## Smoke test on a synthetic clip

`scripts/make_demo_video.py` builds a clip of a person standing and then pivoting to horizontal. This checks that the full pipeline works end to end; it is **not** real fall footage and must not be used to claim accuracy.

## Results

Reference run: 2-vCPU cloud CPU, 640x640, YOLOv8n ONNX FP32, 130-frame synthetic clip.

| Stage | Mean latency |
|---|---|
| Pose (ONNX Runtime) | 81.8 ms |
| Segmentation (ONNX Runtime) | 117.6 ms |
| Pipeline FPS with pose + segmentation | 4.4 |

On the synthetic fall clip, the fall event was raised at t = 3.64 s against a true onset of 3.0 s (within the 2 s tolerance). This clip only verifies the pipeline end to end. It is not real fall footage, and no accuracy claim is made. Evaluate on real labeled video with `evaluate.py` before drawing conclusions.

## Limitations and next steps

- Rule-based fall logic can fail with heavy occlusion or camera angles that hide the torso; a learned temporal classifier over keypoint sequences (e.g. a small LSTM / 1D-CNN) is the natural upgrade.
- Add appearance Re-ID features to keep IDs through long occlusions.
- INT8 quantization / TensorRT for edge devices; process segmentation every N frames to save compute.

## Licensing note

Model weights and export tooling come from [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0. Check its license terms before any commercial use.
