import numpy as np

from evaluate import match_events
from visionguard.events import EventConfig, EventEngine, torso_angle

SQUARE = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.float32)


def kp(upright=True):
    k = np.zeros((17, 3), np.float32)
    k[:, 2] = 0.9
    if upright:   # shoulders above hips
        k[5], k[6], k[11], k[12] = [190, 120, .9], [210, 120, .9], [190, 200, .9], [210, 200, .9]
    else:         # shoulders level with hips, spread horizontally
        k[5], k[6], k[11], k[12] = [120, 200, .9], [120, 215, .9], [220, 200, .9], [220, 215, .9]
    return k


def test_torso_angle():
    assert torso_angle(kp(True)) < 10
    assert torso_angle(kp(False)) > 80
    assert torso_angle(np.zeros((17, 3), np.float32)) is None


def test_zone_intrusion_and_loitering():
    eng = EventEngine(10, SQUARE, EventConfig(loiter_s=2.0))
    types = []
    for f in range(40):
        box = np.array([150, 150, 200, 280]) if f >= 5 else np.array([500, 150, 550, 280])
        types += [e["type"] for e in eng.update(f, [(1, box, .9, kp(True))])]
    assert types.count("intrusion") == 1 and types.count("loitering") == 1


def test_fall_requires_transition_and_hold():
    eng = EventEngine(10, None, EventConfig(fall_hold_s=0.5))
    types = []
    for f in range(30):
        k, box = (kp(True), np.array([150, 100, 250, 300])) if f < 15 else (kp(False), np.array([100, 180, 260, 240]))
        types += [e["type"] for e in eng.update(f, [(1, box, .9, k)])]
    assert types.count("fall") == 1
    # a person already lying (never upright) must not trigger a fall
    eng2, t2 = EventEngine(10, None), []
    for f in range(30):
        t2 += [e["type"] for e in eng2.update(f, [(2, np.array([100, 180, 260, 240]), .9, kp(False))])]
    assert "fall" not in t2


def test_crowding_edge_triggered():
    eng = EventEngine(10, SQUARE, EventConfig(crowd_n=3))
    boxes = [np.array([110 + 30 * i, 150, 130 + 30 * i, 250]) for i in range(3)]
    types = []
    for f in range(10):
        types += [e["type"] for e in eng.update(f, [(i + 1, b, .9, kp(True)) for i, b in enumerate(boxes)])]
    assert types.count("crowding") == 1


def test_event_evaluation_metrics():
    gt = [{"type": "fall", "t": 3.0}, {"type": "intrusion", "t": 1.0}]
    pred = [{"type": "fall", "t": 3.4}, {"type": "fall", "t": 9.0}, {"type": "intrusion", "t": 5.0}]
    r = match_events(pred, gt, tol=1.0)
    assert r["fall"]["tp"] == 1 and r["fall"]["fp"] == 1
    assert r["intrusion"]["tp"] == 0 and r["intrusion"]["fn"] == 1
