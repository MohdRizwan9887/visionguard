"""Create a synthetic demo clip: a person stands, then pivots about their feet to horizontal (simulated fall).

This only smoke-tests the pipeline logic end to end. It is NOT real fall footage: evaluate on real video
(e.g. public fall-detection datasets) before making any accuracy claim.

Usage: python scripts/make_demo_video.py <image.jpg> <out.mp4> <gt.json> [x1,y1,x2,y2 crop of one standing person]
"""
import json
import sys

import cv2
import numpy as np

src, out_path, gt_path = sys.argv[1:4]
crop = tuple(map(int, sys.argv[4].split(","))) if len(sys.argv) > 4 else (700, 0, 1180, 720)  # zidane.jpg
W, H, FPS = 960, 540, 25
img = cv2.imread(src)
x1, y1, x2, y2 = crop
person = img[y1:y2, x1:x2]
s = 0.72
person = cv2.resize(person, None, fx=s, fy=s)
ph, pw = person.shape[:2]

feet = (200, 510)                       # pivot point (feet) on the canvas
n_stand, n_fall, n_hold = 75, 10, 45    # frames standing / falling / lying
canvas = np.full((H, W, 3), 114, np.uint8)
ox, oy = feet[0] - pw // 2, feet[1] - ph
canvas[max(oy, 0):oy + ph, ox:ox + pw] = person[max(-oy, 0):, :]
vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for t in range(n_stand + n_fall + n_hold):
    ang = 0.0 if t < n_stand else 90.0 * min(1.0, (t - n_stand + 1) / n_fall)
    M = cv2.getRotationMatrix2D(feet, -ang, 1.0)           # negative = clockwise: head swings to the right
    M[0, 2] += 2 * np.sin(t / 6)                            # slight jitter
    vw.write(cv2.warpAffine(canvas, M, (W, H), borderValue=(114, 114, 114)))
vw.release()
json.dump([{"type": "fall", "t": n_stand / FPS}], open(gt_path, "w"))
print("wrote", out_path, "| fall starts at t =", n_stand / FPS, "s")
