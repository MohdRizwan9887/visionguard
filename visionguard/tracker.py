"""SORT-style tracker (Kalman filter + Hungarian IoU matching) that carries a per-detection payload (e.g. keypoints)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def box_to_z(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)]).reshape(4, 1)


def x_to_box(x):
    s, r = max(float(x[2, 0]), 1e-6), max(float(x[3, 0]), 1e-6)
    w = np.sqrt(s * r); h = s / w
    return np.array([x[0, 0] - w / 2, x[1, 0] - h / 2, x[0, 0] + w / 2, x[1, 0] + h / 2])


class Track:
    _next_id = 1

    def __init__(self, box, score, payload):
        self.F = np.eye(7); self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7)); self.H[:4, :4] = np.eye(4)
        self.R = np.diag([1.0, 1.0, 10.0, 0.01])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.Q = np.diag([1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 1e-4])
        self.x = np.zeros((7, 1)); self.x[:4] = box_to_z(box)
        self.id = Track._next_id; Track._next_id += 1
        self.score, self.payload = float(score), payload
        self.hits, self.time_since_update = 1, 0

    def predict(self):
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return x_to_box(self.x)

    def update(self, box, score, payload):
        y = box_to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.score, self.payload = float(score), payload
        self.hits += 1; self.time_since_update = 0

    @property
    def box(self):
        return x_to_box(self.x)


class SORTTracker:
    def __init__(self, max_age=25, min_hits=3, iou_threshold=0.3):
        self.max_age, self.min_hits, self.iou_threshold = max_age, min_hits, iou_threshold
        self.tracks: list[Track] = []
        self.frame_count = 0

    def update(self, boxes: np.ndarray, scores: np.ndarray, payloads: list | None = None):
        """Returns list of (track_id, box_xyxy, score, payload) for confirmed, currently matched tracks."""
        self.frame_count += 1
        payloads = payloads if payloads is not None else [None] * len(boxes)
        preds = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        ok = np.isfinite(preds).all(1)
        self.tracks = [t for t, v in zip(self.tracks, ok) if v]; preds = preds[ok]
        iou = iou_matrix(preds, boxes) if len(boxes) else np.zeros((len(preds), 0))
        matched = set()
        if iou.size:
            for r, c in zip(*linear_sum_assignment(-iou)):
                if iou[r, c] >= self.iou_threshold:
                    self.tracks[r].update(boxes[c], scores[c], payloads[c]); matched.add(c)
        for j in range(len(boxes)):
            if j not in matched:
                self.tracks.append(Track(boxes[j], scores[j], payloads[j]))
        out = [(t.id, t.box, t.score, t.payload) for t in self.tracks
               if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame_count <= self.min_hits)]
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return out

    @staticmethod
    def reset_ids():
        Track._next_id = 1
