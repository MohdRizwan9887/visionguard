"""YOLOv8-pose and YOLOv8-seg wrappers on ONNX Runtime (decoding, NMS and mask reconstruction in NumPy/OpenCV)."""
from __future__ import annotations

import time

import cv2
import numpy as np

from .common import cxcywh_to_xyxy_orig, make_session, nms, to_blob

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


class PoseDetector:
    """Person detection + 17 COCO keypoints. Returns boxes (N,5)=[x1,y1,x2,y2,score], kpts (N,17,3)=[x,y,conf]."""

    def __init__(self, path: str, conf: float = 0.35, iou: float = 0.5):
        self.sess = make_session(path)
        self.inp = self.sess.get_inputs()[0].name
        shp = self.sess.get_inputs()[0].shape
        self.size = int(shp[2]) if isinstance(shp[2], int) else 640
        self.conf, self.iou = conf, iou
        self.infer_ms: list[float] = []

    def __call__(self, frame: np.ndarray):
        blob, r, pad = to_blob(frame, self.size)
        t0 = time.perf_counter()
        out = self.sess.run(None, {self.inp: blob})[0]
        self.infer_ms.append((time.perf_counter() - t0) * 1e3)
        p = out[0].T                                   # (8400, 56): box4, conf, 17*3 keypoints
        keep = p[:, 4] >= self.conf
        p = p[keep]
        if len(p) == 0:
            return np.zeros((0, 5), np.float32), np.zeros((0, 17, 3), np.float32)
        boxes = cxcywh_to_xyxy_orig(p[:, :4], r, pad, frame.shape)
        scores = p[:, 4]
        idx = nms(boxes, scores, self.conf, self.iou)
        p, boxes, scores = p[idx], boxes[idx], scores[idx]
        k = p[:, 5:].reshape(-1, 17, 3).copy()
        k[:, :, 0] = (k[:, :, 0] - pad[0]) / r
        k[:, :, 1] = (k[:, :, 1] - pad[1]) / r
        return np.concatenate([boxes, scores[:, None]], 1).astype(np.float32), k.astype(np.float32)


class SegDetector:
    """Instance segmentation. Returns dets (N,6)=[x1,y1,x2,y2,score,cls] and boolean masks (N,H,W)."""

    def __init__(self, path: str, conf: float = 0.35, iou: float = 0.5, classes=(0,)):
        self.sess = make_session(path)
        self.inp = self.sess.get_inputs()[0].name
        shp = self.sess.get_inputs()[0].shape
        self.size = int(shp[2]) if isinstance(shp[2], int) else 640
        self.conf, self.iou, self.classes = conf, iou, classes
        self.infer_ms: list[float] = []

    def __call__(self, frame: np.ndarray):
        H, W = frame.shape[:2]
        blob, r, pad = to_blob(frame, self.size)
        t0 = time.perf_counter()
        out0, protos = self.sess.run(None, {self.inp: blob})[:2]   # (1,116,8400), (1,32,160,160)
        self.infer_ms.append((time.perf_counter() - t0) * 1e3)
        p = out0[0].T
        cls_scores = p[:, 4:84]
        cls = cls_scores.argmax(1)
        scores = cls_scores[np.arange(len(cls)), cls]
        keep = (scores >= self.conf) & np.isin(cls, self.classes)
        p, cls, scores = p[keep], cls[keep], scores[keep]
        empty = (np.zeros((0, 6), np.float32), np.zeros((0, H, W), bool))
        if len(p) == 0:
            return empty
        boxes = cxcywh_to_xyxy_orig(p[:, :4], r, pad, frame.shape)
        idx = nms(boxes, scores, self.conf, self.iou)
        if len(idx) == 0:
            return empty
        p, cls, scores, boxes = p[idx], cls[idx], scores[idx], boxes[idx]

        # mask = sigmoid(coeffs @ prototypes), upsampled, un-letterboxed, cropped to its box
        c, mh, mw = protos[0].shape
        m = 1 / (1 + np.exp(-(p[:, 84:] @ protos[0].reshape(c, -1)))).reshape(-1, mh, mw)
        left, top = pad
        nh, nw = int(round(H * r)), int(round(W * r))
        masks = np.zeros((len(m), H, W), bool)
        for i, mi in enumerate(m):
            full = cv2.resize(mi, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            full = full[top:top + nh, left:left + nw]
            full = cv2.resize(full, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
            x1, y1, x2, y2 = boxes[i].astype(int)
            box_mask = np.zeros((H, W), bool)
            box_mask[y1:y2 + 1, x1:x2 + 1] = True
            masks[i] = full & box_mask
        dets = np.concatenate([boxes, scores[:, None], cls[:, None]], 1).astype(np.float32)
        return dets, masks
