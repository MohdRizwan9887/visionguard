"""Shared pre/post-processing helpers for ONNX YOLOv8 models."""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(img: np.ndarray, size: int = 640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, (left, top)


def make_session(path: str, providers=None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=providers or ["CPUExecutionProvider"])


def to_blob(frame: np.ndarray, size: int):
    img, r, pad = letterbox(frame, size)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (size, size), swapRB=True)
    return blob.astype(np.float32), r, pad


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, conf: float, iou: float) -> np.ndarray:
    if len(scores) == 0:
        return np.zeros((0,), dtype=int)
    xywh = np.stack([boxes_xyxy[:, 0], boxes_xyxy[:, 1],
                     boxes_xyxy[:, 2] - boxes_xyxy[:, 0], boxes_xyxy[:, 3] - boxes_xyxy[:, 1]], 1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, iou)
    return np.array(idx).reshape(-1).astype(int)


def cxcywh_to_xyxy_orig(b: np.ndarray, r: float, pad, shape):
    """Map letterboxed cx,cy,w,h boxes back to original-image x1,y1,x2,y2."""
    x, y, w, h = b.T
    H, W = shape[:2]
    x1 = np.clip((x - w / 2 - pad[0]) / r, 0, W - 1)
    y1 = np.clip((y - h / 2 - pad[1]) / r, 0, H - 1)
    x2 = np.clip((x + w / 2 - pad[0]) / r, 0, W - 1)
    y2 = np.clip((y + h / 2 - pad[1]) / r, 0, H - 1)
    return np.stack([x1, y1, x2, y2], 1)
