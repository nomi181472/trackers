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

        self.rng = np.random.default_rng(self.seed)
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
                amp = H * (0.02 + 0.03 * rng.random())
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
            amp = H * 0.02
            k = 1.0 + 0.3 * idx
            arr = np.zeros((T, 2))
            for t in range(T):
                s = t / (T - 1)
                arr[t, 0] = x0 + (x1 - x0) * s
                arr[t, 1] = (lane + amp * math.sin(k * s * math.tau)) * H
            paths[idx] = arr
        # clamp every path inside the frame
        for idx in paths:
            paths[idx][:, 0] = np.clip(paths[idx][:, 0], r + 2, W - r - 2)
            paths[idx][:, 1] = np.clip(paths[idx][:, 1], r + 2, H - r - 2)
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
        radii = [r, r, max(20, r - 3), r + 3][: max(self.num_objects, 1)]
        radii = (radii * self.num_objects)[: self.num_objects]

        shake_dx = rng.integers(-self.shake_px, self.shake_px + 1, size=T) if self.camera_shake else None
        shake_dy = rng.integers(-self.shake_px, self.shake_px + 1, size=T) if self.camera_shake else None

        composed_all = []
        for t in range(T):
            frame = np.full((H, W, 3), SKY, dtype=np.uint8)
            entries = []
            pts = []
            for i in range(self.num_objects):
                cx, cy = paths[i][t]
                rr = radii[i]
                # ground shadow first
                cv_ell = None
                pts.append((int(cx), int(cy), rr))
            # draw in two passes: shadows then balls
            for (cx, cy, rr) in pts:
                cv = _cv()
                cv.ellipse(frame, (int(cx), int(cy + rr * 0.85)), (int(rr * 0.95), int(rr * 0.35)),
                           0, 0, 360, (196, 199, 205), -1)
            for i in range(self.num_objects):
                cx, cy = paths[i][t]
                rr = radii[i]
                # ball with a soft highlight
                cv = _cv()
                cv.circle(frame, (int(cx), int(cy)), rr, colors[i], -1, cv.LINE_AA)
                cv.circle(frame, (int(cx - rr * 0.3), int(cy - rr * 0.3)), max(3, rr // 4),
                          (min(255, colors[i][0] + 70), min(255, colors[i][1] + 70),
                           min(255, colors[i][2] + 70)), -1, cv.LINE_AA)

                occluded = bool(wall and wall[0] <= cx <= wall[2])
                box = [int(cx - rr), int(cy - rr), int(cx + rr), int(cy + rr)]
                entries.append(dict(
                    id=i, box=box, visible=not occluded, occluded=occluded,
                    center=(float(cx), float(cy)),
                    color=list(colors[i]),
                    radius=rr,
                ))
            if wall:
                cv = _cv()
                cv.rectangle(frame, (wall[0], 0), (wall[2], H), WALL_COLOR, -1)
                cv.rectangle(frame, (wall[0], 0), (wall[2], H),
                             (44, 46, 60), 2)
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
            crossing=self.crossing, occlusion=self.occlusion, blur=self.blur,
            camera_shake=self.camera_shake, similar_colors=self.similar_colors,
            occluder_box=wall, seed=self.seed,
        )

    # ------------------------------------------------------------------ #
    def render_annotated_frame(self, t: int, tracks=None, draw_gt=True, labels=True):
        """Frame with optional tracker boxes (list of Track) and GT overlay."""
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
        return img


def _cv():
    import cv2
    return cv2