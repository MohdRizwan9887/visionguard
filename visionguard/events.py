"""Rule-based safety event engine on top of tracked people: zone intrusion, loitering, crowding, fall detection."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


@dataclass
class EventConfig:
    loiter_s: float = 5.0            # dwell time inside zone to raise "loitering"
    crowd_n: int = 3                 # people inside zone to raise "crowding"
    fall_angle_deg: float = 60.0     # torso angle from vertical considered horizontal
    upright_angle_deg: float = 35.0  # torso angle considered upright
    fall_hold_s: float = 0.5         # must stay horizontal this long
    fall_lookback_s: float = 2.0     # must have been upright within this window (transition, not just lying)
    kpt_conf: float = 0.3


def torso_angle(kpts: np.ndarray, thr: float = 0.3):
    """Angle (deg) between the shoulder->hip axis and vertical. 0 = upright, 90 = horizontal. None if not visible."""
    sh = [kpts[i, :2] for i in (L_SH, R_SH) if kpts[i, 2] >= thr]
    hp = [kpts[i, :2] for i in (L_HIP, R_HIP) if kpts[i, 2] >= thr]
    if not sh or not hp:
        return None
    s, h = np.mean(sh, 0), np.mean(hp, 0)
    return math.degrees(math.atan2(abs(s[0] - h[0]), abs(s[1] - h[1]) + 1e-6))


def point_in_zone(poly_px: np.ndarray | None, pt) -> bool:
    if poly_px is None:
        return True
    return cv2.pointPolygonTest(poly_px.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0


class EventEngine:
    def __init__(self, fps: float, zone_px: np.ndarray | None = None, cfg: EventConfig | None = None):
        self.fps, self.zone, self.cfg = fps, zone_px, cfg or EventConfig()
        self.state: dict[int, dict] = {}
        self.crowd_active = False

    def _st(self, tid):
        return self.state.setdefault(tid, dict(in_zone=False, enter_f=None, loiter=False, last_upright=-10**9,
                                               horiz=0, fallen=False))

    def update(self, frame_idx: int, tracks) -> list[dict]:
        """tracks: iterable of (track_id, box_xyxy, score, kpts (17,3) or None). Returns list of event dicts."""
        cfg, ev, t = self.cfg, [], frame_idx / self.fps
        inside_ids = []
        for tid, box, score, kpts in tracks:
            st = self._st(tid)
            foot = ((box[0] + box[2]) / 2, box[3])
            inside = point_in_zone(self.zone, foot)
            if inside:
                inside_ids.append(tid)
            if inside and not st["in_zone"]:
                st["in_zone"], st["enter_f"], st["loiter"] = True, frame_idx, False
                if self.zone is not None:
                    ev.append(dict(t=round(t, 2), frame=frame_idx, type="intrusion", track_id=int(tid)))
            elif not inside:
                st["in_zone"], st["enter_f"], st["loiter"] = False, None, False
            if inside and st["enter_f"] is not None and not st["loiter"] \
                    and (frame_idx - st["enter_f"]) / self.fps >= cfg.loiter_s:
                st["loiter"] = True
                ev.append(dict(t=round(t, 2), frame=frame_idx, type="loitering", track_id=int(tid)))

            # ---- fall detection from pose ----
            ang = torso_angle(kpts, cfg.kpt_conf) if kpts is not None else None
            w, h = box[2] - box[0], box[3] - box[1]
            horizontal = (ang is not None and ang >= cfg.fall_angle_deg) or (ang is None and w / max(h, 1) > 1.3)
            upright = ang is not None and ang <= cfg.upright_angle_deg
            if upright:
                st["last_upright"], st["horiz"], st["fallen"] = frame_idx, 0, False
            elif horizontal:
                st["horiz"] += 1
                recent = (frame_idx - st["last_upright"]) / self.fps <= cfg.fall_lookback_s
                if st["horiz"] >= math.ceil(cfg.fall_hold_s * self.fps) and recent and not st["fallen"]:
                    st["fallen"] = True
                    ev.append(dict(t=round(t, 2), frame=frame_idx, type="fall", track_id=int(tid)))
            else:
                st["horiz"] = 0

        n_in = len(inside_ids)
        if n_in >= cfg.crowd_n and not self.crowd_active:
            self.crowd_active = True
            ev.append(dict(t=round(t, 2), frame=frame_idx, type="crowding", count=n_in))
        elif n_in < cfg.crowd_n:
            self.crowd_active = False
        return ev
