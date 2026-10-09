"""Synthetic scene generator.

Renders a short clip of independently-moving coloured balls with known
ground truth, plus a menu of "press here to break a tracker" perturbations:

  * occlusion   -- a static wall the balls pass *behind* (box disappears)
  * crossing    -- two balls swap sides, forcing a physical overlap at the
                   exact moment their IDs become ambiguous
  * camera shake-- a global translation every frame (GMC stress)
  * blur        -- motion blur that saps detector confidence
  * identity disguise -- all balls the same colour (no appearance cues)

The beauty of synthetic scenes: we *know* the truth, so every failure the
displayed tracker shows can be blamed precisely on the right cause.
"""
from __future__ import annotations

import math

import numpy as np

BALL_COLORS = [
    (235, 64, 52), (250, 213, 58), (75, 191, 123), (64, 158, 235),
    (178, 93, 235), (235, 140, 88), (60, 210, 215), (240, 96, 160),
]
WALL_COLOR = (60, 62, 78)
SKY = (245, 245, 245)
GROUND = (225, 228, 233)


def _as_bool(val, default=False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes", "on")
    return bool(val)


class Scenario:
    def __init__(self, params: dict):
        self.params = dict(params)
        self.seed = int(params.get("seed", 7))
        self.fps = int(params.get("fps", 15))
        self.duration = float(params.get("duration_seconds", 8.0))
        self.width = int(params.get("width", 640))
        self.height = int(params.get("height", 640))
        self.num_objects = int(params.get("num_objects", 3))
        self.crossing = _as_bool(params.get("crossing", True), True)
        self.occlusion = _as_bool(params.get("occlusion", True), True)
        self.occluder_width = int(float(params.get("occluder_width", 56)))
        self.camera_shake = _as_bool(params.get("camera_shake", False), False)
        self.shake_px = int(float(params.get("shake_px", 8)))
        self.blur = _as_bool(params.get("blur", False), False)
        self.blur_sigma = float(params.get("blur_sigma", 3.0))
        self.similar_colors = _as_bool(params.get("similar_colors", False), False)
        self.noise = float(params.get("noise", 0.0))
        obj_type = str(params.get("object_type", "ball")).lower()
        if obj_type == "skeleton":
            obj_type = "person"
        elif obj_type not in ("ball", "person", "car"):
            obj_type = "ball"
        self.object_type = obj_type

        self.rng = np.random.default_rng(self.seed)
        if "frames" in params and params["frames"] is not None:
            self.n_frames = max(2, int(params["frames"]))
        else:
            self.n_frames = max(2, int(self.duration * self.fps))
        self.radius = self.height // 24

        self.frames = None       # np.uint8 (T,H,W,3) BGR
        self.gt = None           # list per frame of {"id","box","visible","occluded","center","color"}
        self.occluder_box = None  # [x1,y1,x2,y2] or None
        self.meta: dict = {}
        self._build()

    # ------------------------------------------------------------------ #
    def _paths(self):
        """Index by object id: (cx(t), cy(t)) arrays."""
        W, H, T = self.width, self.height, self.n_frames
        r = self.radius
        rng = self.rng
        paths = {}
        n = self.num_objects
        if self.crossing and n >= 2:
            # Diagonal pair that meets exactly at the centre at T/2.
            p0 = (W * 0.18, H * 0.32)
            p1 = (W * 0.82, H * 0.68)
            def seg(a, b):
                return lambda s: (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)
            paths[0] = np.array([seg(p0, p1)(s) for s in np.linspace(0, 1, T)])
            paths[1] = np.array([seg(p1, p0)(s) for s in np.linspace(0, 1, T)])
        else:
            # Independent lanes, gentle wobble, no designed collision.
            for i in range(n):
                lane = 0.2 + 0.2 * (i % 3) + 0.06 * rng.uniform(-1, 1)
                x0 = W * (0.1 + 0.06 * i)
                x1 = W * (0.9 - 0.06 * i)
                amp = 0.02 + 0.03 * rng.random()
                k = 1 + 0.5 * i
                ph = rng.uniform(0, math.tau)
                arr = np.zeros((T, 2))
                for t in range(T):
                    s = t / (T - 1)
                    arr[t, 0] = x0 + (x1 - x0) * s
                    arr[t, 1] = (lane + amp * math.sin(k * s * math.tau + ph)) * H
                paths[i] = arr
        # Any remaining objects get simple parallel lanes too.
        for i in range(n - len(paths)):
            idx = len(paths)
            lane = 0.25 + 0.17 * idx
            x0 = W * (0.12 + 0.04 * idx)
            x1 = W * (0.88 - 0.04 * idx)
            amp = 0.02
            k = 1.0 + 0.3 * idx
            arr = np.zeros((T, 2))
            for t in range(T):
                s = t / (T - 1)
                arr[t, 0] = x0 + (x1 - x0) * s
                arr[t, 1] = (lane + amp * math.sin(k * s * math.tau)) * H
            paths[idx] = arr

        # clamp every path inside the frame based on object type
        if self.object_type == "person":
            clamp_x = max(r + 2, int(r * 0.9) + 4)
            clamp_y = max(r + 2, int(r * 1.6) + 4)
        elif self.object_type == "car":
            clamp_x = max(r + 2, int(r * 1.8) + 4)
            clamp_y = max(r + 2, int(r * 0.9) + 4)
        else:
            clamp_x = r + 2
            clamp_y = r + 2

        for idx in paths:
            paths[idx][:, 0] = np.clip(paths[idx][:, 0], clamp_x, W - clamp_x)
            paths[idx][:, 1] = np.clip(paths[idx][:, 1], clamp_y, H - clamp_y)
        return paths

    def _build(self):
        W, H, T = self.width, self.height, self.n_frames
        r = self.radius
        rng = self.rng

        # static occluder wall, right-of-centre
        wall = None
        if self.occlusion:
            cx = int(W * 0.58)
            halfw = max(18, self.occluder_width // 2)
            wall = [cx - halfw, -r, cx + halfw, H + r]
        self.occluder_box = wall

        paths = self._paths()
        colors = {}
        for i in range(self.num_objects):
            colors[i] = (BALL_COLORS[i % len(BALL_COLORS)] if not self.similar_colors
                         else BALL_COLORS[2])

        frames = np.zeros((T, H, W, 3), dtype=np.uint8)
        frames[:, :, :] = SKY
        gt = [[] for _ in range(T)]

        # Object dimensions per object index
        if self.object_type == "person":
            scales = [1.0, 1.0, 0.92, 1.08]
            dims = []
            for i in range(self.num_objects):
                sc = scales[i % len(scales)]
                hh = max(24, int(r * 1.55 * sc))
                hw = max(10, int(hh * 0.38))
                dims.append((hw, hh))
        elif self.object_type == "car":
            scales = [1.0, 1.0, 0.92, 1.08]
            dims = []
            for i in range(self.num_objects):
                sc = scales[i % len(scales)]
                hw = max(24, int(r * 1.65 * sc))
                hh = max(12, int(hw * 0.48))
                dims.append((hw, hh))
        else:
            radii = [r, r, max(20, r - 3), r + 3][: max(self.num_objects, 1)]
            radii = (radii * self.num_objects)[: self.num_objects]
            dims = [(rr, rr) for rr in radii]

        shake_dx = rng.integers(-self.shake_px, self.shake_px + 1, size=T) if self.camera_shake else None
        shake_dy = rng.integers(-self.shake_px, self.shake_px + 1, size=T) if self.camera_shake else None

        for t in range(T):
            frame = np.full((H, W, 3), SKY, dtype=np.uint8)
            entries = []

            # Pass 1: Ground shadows
            cv = _cv()
            for i in range(self.num_objects):
                cx, cy = paths[i][t]
                hw, hh = dims[i]
                if self.object_type == "person":
                    cv.ellipse(frame, (int(cx), int(cy + hh * 0.96)),
                               (int(hw * 1.1), max(3, int(hh * 0.12))),
                               0, 0, 360, (196, 199, 205), -1)
                elif self.object_type == "car":
                    cv.ellipse(frame, (int(cx), int(cy + hh * 0.92)),
                               (int(hw * 0.95), max(4, int(hh * 0.22))),
                               0, 0, 360, (196, 199, 205), -1)
                else:
                    cv.ellipse(frame, (int(cx), int(cy + hh * 0.96)),
                               (int(hw * 0.95), max(2, int(hh * 0.18))),
                               0, 0, 360, (196, 199, 205), -1)

            # Pass 2: Objects
            for i in range(self.num_objects):
                cx, cy = paths[i][t]
                hw, hh = dims[i]
                col = colors[i]

                # Motion direction for orientation
                if t < T - 1:
                    vx = paths[i][t + 1][0] - paths[i][t][0]
                elif t > 0:
                    vx = paths[i][t][0] - paths[i][t - 1][0]
                else:
                    vx = 1.0
                dir_sign = 1 if vx >= 0 else -1

                if self.object_type == "person":
                    _draw_person_skeleton(cv, frame, cx, cy, hw, hh, col, t, dir_sign, i)
                elif self.object_type == "car":
                    _draw_car(cv, frame, cx, cy, hw, hh, col, dir_sign)
                else:
                    _draw_ball(cv, frame, cx, cy, hw, col)

                occluded = bool(wall and wall[0] <= cx <= wall[2])
                box = [int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)]
                entries.append(dict(
                    id=i, box=box, visible=not occluded, occluded=occluded,
                    center=(float(cx), float(cy)),
                    color=list(colors[i]),
                    radius=hw,
                ))

            if wall:
                cv.rectangle(frame, (wall[0], 0), (wall[2], H), WALL_COLOR, -1)
                cv.rectangle(frame, (wall[0], 0), (wall[2], H), (44, 46, 60), 2)
                cv.line(frame, (wall[0], 0), (wall[2], H), (150, 152, 166), 1)

            # camera shake: warp scene *and* ground truth together
            if shake_dx is not None:
                dx, dy = int(shake_dx[t]), int(shake_dy[t])
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                frame = _cv().warpAffine(frame, M, (W, H), borderValue=SKY)
                for e in entries:
                    e["box"] = [int(np.clip(e["box"][0] + dx, 0, W)),
                                int(np.clip(e["box"][1] + dy, 0, H)),
                                int(np.clip(e["box"][2] + dx, 0, W)),
                                int(np.clip(e["box"][3] + dy, 0, H))]

            if self.blur and self.blur_sigma > 0:
                frame = _cv().GaussianBlur(frame, (0, 0), self.blur_sigma)

            if self.noise > 0:
                nz = rng.normal(0, self.noise * 255, frame.shape).astype(np.float32)
                frame = np.clip(frame.astype(np.float32) + nz, 0, 255).astype(np.uint8)

            frames[t] = frame
            gt[t] = entries

        self.frames = frames
        self.gt = gt
        self.meta = dict(
            fps=self.fps, width=W, height=H, frames=T, num_objects=self.num_objects,
            object_type=self.object_type,
            crossing=self.crossing, occlusion=self.occlusion, blur=self.blur,
            camera_shake=self.camera_shake, similar_colors=self.similar_colors,
            occluder_box=wall, seed=self.seed,
        )

    # ------------------------------------------------------------------ #
    # Single-object target                                                 #
    # ------------------------------------------------------------------ #

    def target_id(self) -> int:
        """Which object a single-object follower is asked to follow.

        One rule, owned here, so the runner (which seeds the follower) and the
        scorer (which grades it) can never disagree about the subject.
        """
        return 0

    def first_visible_target_frame(self) -> int | None:
        """First frame the target is actually visible, or None if never.

        Seeding a follower on a box that is hidden behind the wall is a
        guaranteed loss, so the runner waits for a frame we can actually see.
        """
        tid = self.target_id()
        for t, entries in enumerate(self.gt):
            for e in entries:
                if e["id"] == tid and e["visible"]:
                    return t
        return None

    # ------------------------------------------------------------------ #
    def _draw_hud(self, img, tracker_name: str, frame_no: str) -> None:
        """Tracker name on the left, frame number on the right.

        Drawn as filled chips rather than bare text: the scene background is
        light sky, so white-on-nothing would be unreadable.  The name is clipped
        so a long one can never run into the frame counter.
        """
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.5, 1
        pad = 5

        def text_w(text):
            (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
            return tw

        def chip(text, x0, from_left):
            (tw, th), base = cv2.getTextSize(text, font, scale, thick)
            w = tw + 2 * pad
            x = x0 if from_left else x0 - w
            cv2.rectangle(img, (x, 2), (x + w, 2 + th + base + 6), (30, 30, 38), -1)
            cv2.putText(img, text, (x + pad, 2 + th + 3), font, scale,
                        (255, 255, 255), thick, cv2.LINE_AA)

        if tracker_name:
            room = img.shape[1] - 2 * pad - (text_w(frame_no) + 2 * pad) - 8
            if text_w(tracker_name) > room:  # ellipsise rather than overlap
                name = tracker_name
                while name and text_w(name + "...") > room:
                    name = name[:-1]
                tracker_name = name + "..."
            chip(tracker_name, 2, True)
        chip(frame_no, img.shape[1] - 2, False)

    # ------------------------------------------------------------------ #
    def render_annotated_frame(self, t: int, tracks=None, draw_gt=True, labels=True,
                               tracker_name: str = "", show_frame_no: bool = True):
        """Frame with optional tracker boxes (list of Track) and GT overlay.

        `tracker_name` and `show_frame_no` add the corner HUD used on the
        per-tracker videos; callers that draw their own banner on top (event
        thumbnails) switch the frame number off.
        """
        import cv2
        img = self.frames[t].copy()
        gt = self.gt[t]
        if draw_gt:
            for e in gt:
                if e["visible"]:
                    x1, y1, x2, y2 = e["box"]
                    # light GT outline
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0, ), 1)
        if tracks:
            from app.core.trackers import color_for
            for tr in tracks:
                x1, y1, x2, y2 = tr.box
                col = color_for(tr.id)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                if labels:
                    cv2.rectangle(img, (x1, max(0, y1 - 16)), (x1 + 70, max(0, y1)), col, -1)
                    cv2.putText(img, f"ID {tr.id}", (x1 + 3, max(13, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        if self.occluder_box and draw_gt:
            x1, _, x2, _ = self.occluder_box
            cv2.line(img, (x1, 0), (x1, self.height), (200, 60, 60), 1, cv2.LINE_AA)  # hidden issue marker
        # 0-based, so it lines up with the `frame` number on every reported event
        if tracker_name or show_frame_no:
            self._draw_hud(img, tracker_name, f"f {t}/{self.n_frames}")
        return img


def _cv():
    import cv2
    return cv2


def _draw_ball(cv, frame, cx, cy, rr, col):
    cv.circle(frame, (int(cx), int(cy)), rr, col, -1, cv.LINE_AA)
    dark = (max(0, col[0] - 45), max(0, col[1] - 45), max(0, col[2] - 45))
    cv.ellipse(frame, (int(cx), int(cy)), (max(2, int(rr * 0.6)), rr), 25, 0, 360, dark, max(1, rr // 10), cv.LINE_AA)
    cv.circle(frame, (int(cx - rr * 0.3), int(cy - rr * 0.3)), max(2, rr // 4),
              (min(255, col[0] + 70), min(255, col[1] + 70), min(255, col[2] + 70)), -1, cv.LINE_AA)


def _draw_person_skeleton(cv, frame, cx, cy, hw, hh, col, t, dir_sign, i):
    r_head = max(5, int(hh * 0.20))
    head_cy = int(cy - hh + r_head + 2)
    head_cx = int(cx)

    # Head & face
    cv.circle(frame, (head_cx, head_cy), r_head, col, -1, cv.LINE_AA)
    cv.circle(frame, (head_cx - dir_sign * 1, head_cy - 1), max(2, r_head // 3),
              (min(255, col[0] + 60), min(255, col[1] + 60), min(255, col[2] + 60)), -1, cv.LINE_AA)
    cv.circle(frame, (int(head_cx + dir_sign * r_head * 0.45), int(head_cy - r_head * 0.1)),
              max(1, r_head // 4), (255, 255, 255), -1, cv.LINE_AA)

    # Spine & Torso
    neck_y = head_cy + r_head
    pelvis_y = int(cy + hh * 0.12)
    cv.line(frame, (head_cx, neck_y), (head_cx, pelvis_y), col, 3, cv.LINE_AA)

    # Shoulders
    shoulder_y = neck_y + max(2, int(hh * 0.08))
    sh_w = max(6, int(hw * 0.85))
    l_shoulder = (head_cx - sh_w, shoulder_y)
    r_shoulder = (head_cx + sh_w, shoulder_y)
    cv.line(frame, l_shoulder, r_shoulder, col, 2, cv.LINE_AA)

    # Hips
    hip_w = max(4, int(hw * 0.50))
    l_hip = (head_cx - hip_w, pelvis_y)
    r_hip = (head_cx + hip_w, pelvis_y)
    cv.line(frame, l_hip, r_hip, col, 2, cv.LINE_AA)

    # Walking gait cycle
    stride_phase = (t * 0.55 + i * 1.8) % (2 * math.pi)
    leg_swing = math.sin(stride_phase) * (hw * 0.85) * dir_sign
    arm_swing = -leg_swing * 0.8

    knee_y = int(pelvis_y + hh * 0.42)
    ankle_y = int(cy + hh - 3)

    # Left leg
    l_knee_x = int(l_hip[0] + leg_swing * 0.55)
    l_ankle_x = int(l_hip[0] + leg_swing)
    l_ankle_y = ankle_y - int(abs(leg_swing) * 0.18)
    cv.line(frame, l_hip, (l_knee_x, knee_y), col, 3, cv.LINE_AA)
    cv.line(frame, (l_knee_x, knee_y), (l_ankle_x, l_ankle_y), col, 2, cv.LINE_AA)
    cv.line(frame, (l_ankle_x, l_ankle_y), (l_ankle_x + dir_sign * 5, l_ankle_y), (50, 50, 60), 2, cv.LINE_AA)

    # Right leg (opposite phase)
    r_knee_x = int(r_hip[0] - leg_swing * 0.55)
    r_ankle_x = int(r_hip[0] - leg_swing)
    r_ankle_y = ankle_y - int(abs(-leg_swing) * 0.18)
    cv.line(frame, r_hip, (r_knee_x, knee_y), col, 3, cv.LINE_AA)
    cv.line(frame, (r_knee_x, knee_y), (r_ankle_x, r_ankle_y), col, 2, cv.LINE_AA)
    cv.line(frame, (r_ankle_x, r_ankle_y), (r_ankle_x + dir_sign * 5, r_ankle_y), (50, 50, 60), 2, cv.LINE_AA)

    # Arms
    elbow_y = int(shoulder_y + hh * 0.32)
    wrist_y = int(shoulder_y + hh * 0.60)

    l_elbow_x = int(l_shoulder[0] + arm_swing * 0.5)
    l_wrist_x = int(l_shoulder[0] + arm_swing)
    cv.line(frame, l_shoulder, (l_elbow_x, elbow_y), col, 2, cv.LINE_AA)
    cv.line(frame, (l_elbow_x, elbow_y), (l_wrist_x, wrist_y), col, 2, cv.LINE_AA)

    r_elbow_x = int(r_shoulder[0] - arm_swing * 0.5)
    r_wrist_x = int(r_shoulder[0] - arm_swing)
    cv.line(frame, r_shoulder, (r_elbow_x, elbow_y), col, 2, cv.LINE_AA)
    cv.line(frame, (r_elbow_x, elbow_y), (r_wrist_x, wrist_y), col, 2, cv.LINE_AA)

    # Joint markers
    joints = [
        (head_cx, shoulder_y), (head_cx, pelvis_y),
        l_shoulder, r_shoulder,
        (l_elbow_x, elbow_y), (r_elbow_x, elbow_y),
        (l_wrist_x, wrist_y), (r_wrist_x, wrist_y),
        l_hip, r_hip,
        (l_knee_x, knee_y), (r_knee_x, knee_y),
        (l_ankle_x, l_ankle_y), (r_ankle_x, r_ankle_y),
    ]
    for jx, jy in joints:
        cv.circle(frame, (int(jx), int(jy)), 2, (255, 255, 255), -1, cv.LINE_AA)


def _draw_car(cv, frame, cx, cy, hw, hh, col, dir_sign):
    x_left = int(cx - hw * 0.95)
    x_right = int(cx + hw * 0.95)
    y_bottom = int(cy + hh * 0.65)
    y_belt = int(cy + hh * 0.05)
    y_roof = int(cy - hh * 0.85)

    # 1. Main body lower chassis
    cv.rectangle(frame, (x_left, y_belt), (x_right, y_bottom), col, -1)
    cv.ellipse(frame, (x_left + 4, int((y_belt + y_bottom) / 2)), (5, int((y_bottom - y_belt) / 2)), 0, 90, 270, col, -1)
    cv.ellipse(frame, (x_right - 4, int((y_belt + y_bottom) / 2)), (5, int((y_bottom - y_belt) / 2)), 0, 270, 90, col, -1)

    # 2. Cabin / roof (trapezoid)
    rf_rear = int(cx - dir_sign * hw * 0.40)
    rf_front = int(cx + dir_sign * hw * 0.25)
    c_rear = int(cx - dir_sign * hw * 0.70)
    c_front = int(cx + dir_sign * hw * 0.60)

    roof_pts = np.array([[(c_rear, y_belt), (rf_rear, y_roof), (rf_front, y_roof), (c_front, y_belt)]], dtype=np.int32)
    cv.fillPoly(frame, roof_pts, col)

    # Windows
    glass_col = (235, 225, 205)
    win_roof_rear = int(rf_rear + (3 if rf_front > rf_rear else -3))
    win_roof_front = int(rf_front - (3 if rf_front > rf_rear else -3))
    win_base_rear = int(c_rear + (4 if c_front > c_rear else -4))
    win_base_front = int(c_front - (4 if c_front > c_rear else -4))

    win_pts = np.array([[(win_base_rear, y_belt - 2), (win_roof_rear, y_roof + 3),
                         (win_roof_front, y_roof + 3), (win_base_front, y_belt - 2)]], dtype=np.int32)
    cv.fillPoly(frame, win_pts, glass_col)
    cv.line(frame, (int(cx), y_roof + 3), (int(cx), y_belt - 2), (50, 50, 60), 2, cv.LINE_AA)

    # 3. Wheels
    wheel_r = max(4, int(hh * 0.38))
    w1_x = int(cx - hw * 0.52)
    w2_x = int(cx + hw * 0.52)
    wheel_y = int(cy + hh * 0.68)
    cv.circle(frame, (w1_x, wheel_y), wheel_r, (35, 35, 42), -1, cv.LINE_AA)
    cv.circle(frame, (w2_x, wheel_y), wheel_r, (35, 35, 42), -1, cv.LINE_AA)
    cv.circle(frame, (w1_x, wheel_y), max(2, wheel_r // 2), (180, 185, 192), -1, cv.LINE_AA)
    cv.circle(frame, (w2_x, wheel_y), max(2, wheel_r // 2), (180, 185, 192), -1, cv.LINE_AA)

    # 4. Lights
    front_x = x_right if dir_sign == 1 else x_left
    rear_x = x_left if dir_sign == 1 else x_right
    light_y = int(y_belt + (y_bottom - y_belt) * 0.35)
    cv.circle(frame, (front_x, light_y), max(2, int(hh * 0.14)), (120, 240, 255), -1, cv.LINE_AA)
    cv.circle(frame, (rear_x, light_y), max(2, int(hh * 0.12)), (40, 40, 230), -1, cv.LINE_AA)

    cv.line(frame, (x_left + 4, y_belt), (x_right - 4, y_belt),
            (min(255, col[0] + 50), min(255, col[1] + 50), min(255, col[2] + 50)), 1, cv.LINE_AA)