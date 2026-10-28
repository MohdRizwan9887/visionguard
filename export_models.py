"""Export YOLOv8 pose and segmentation models to ONNX (opset 13).

Usage: python export_models.py            # downloads yolov8n-pose.pt / yolov8n-seg.pt if missing
"""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--size", default="n", help="YOLOv8 size: n, s, m ...")
a = ap.parse_args()

for task in ("pose", "seg"):
    path = YOLO(f"models/yolov8{a.size}-{task}.pt").export(format="onnx", imgsz=a.imgsz, opset=13, simplify=True)
    print(task, "->", path)
