"""Event-level evaluation: precision / recall / F1 per event type with a temporal tolerance.

Usage:
    python evaluate.py --pred events.jsonl --gt ground_truth.json --tol 2.0
ground_truth.json: [{"type": "fall", "t": 2.4}, {"type": "intrusion", "t": 0.8}, ...]
"""
import argparse
import json
from collections import defaultdict


def match_events(preds: list[dict], gts: list[dict], tol: float = 2.0) -> dict:
    types = sorted({e["type"] for e in preds} | {e["type"] for e in gts})
    report = {}
    tp_all = fp_all = fn_all = 0
    for ty in types:
        P = sorted([e for e in preds if e["type"] == ty], key=lambda e: e["t"])
        G = sorted([e for e in gts if e["type"] == ty], key=lambda e: e["t"])
        used, tp = set(), 0
        for g in G:  # greedy nearest-in-time matching, each prediction used at most once
            best, best_d = None, tol + 1e-9
            for i, p in enumerate(P):
                d = abs(p["t"] - g["t"])
                if i not in used and d <= best_d:
                    best, best_d = i, d
            if best is not None:
                used.add(best); tp += 1
        fp, fn = len(P) - tp, len(G) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        report[ty] = dict(tp=tp, fp=fp, fn=fn, precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3))
        tp_all += tp; fp_all += fp; fn_all += fn
    p = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    r = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    report["overall"] = dict(tp=tp_all, fp=fp_all, fn=fn_all, precision=round(p, 3), recall=round(r, 3),
                             f1=round(2 * p * r / (p + r), 3) if p + r else 0.0)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tol", type=float, default=2.0, help="temporal tolerance in seconds")
    a = ap.parse_args()
    preds = [json.loads(l) for l in open(a.pred) if l.strip()]
    gts = json.load(open(a.gt))
    for k, v in match_events(preds, gts, a.tol).items():
        print(f"{k:10s} {v}")


if __name__ == "__main__":
    main()
