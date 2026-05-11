import cv2
import numpy as np

CANVAS      = 512
CONF_THRESH = 0.20   # threshold for drawing
CONF_BBOX   = 0.15   # lower threshold used only for bounding-box computation

# MoveNet keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16

# BGR colors
_DARK  = (45,  52,  54)
_BLUE  = (210, 140,  55)   # left side
_RED   = (55,   75, 220)   # right side
_FILL  = (215, 210, 200)   # torso polygon
_WHITE = (255, 255, 255)

CONNECTIONS = [
    # face
    (NOSE, L_EYE, _DARK), (NOSE, R_EYE, _DARK),
    (L_EYE, L_EAR, _DARK), (R_EYE, R_EAR, _DARK),
    # torso
    (L_SH, R_SH, _DARK), (L_SH, L_HIP, _DARK),
    (R_SH, R_HIP, _DARK), (L_HIP, R_HIP, _DARK),
    # left limbs
    (L_SH, L_EL, _BLUE), (L_EL, L_WR, _BLUE),
    (L_HIP, L_KN, _BLUE), (L_KN, L_AN, _BLUE),
    # right limbs
    (R_SH, R_EL, _RED), (R_EL, R_WR, _RED),
    (R_HIP, R_KN, _RED), (R_KN, R_AN, _RED),
]

_LEFT_JOINTS = {L_SH, L_EL, L_WR, L_HIP, L_KN, L_AN}


def _make_px(kps: np.ndarray):
    """
    Return a (y_norm, x_norm) → (canvas_x, canvas_y) mapper that scales and
    centers the skeleton so it fills ~80% of the canvas.

    Uses a slightly lower confidence threshold (CONF_BBOX) so extremities are
    included in the bounding box even if they fall below the drawing threshold.
    """
    vis = [i for i in range(17) if kps[i, 2] >= CONF_BBOX]

    if len(vis) >= 2:
        ys = kps[vis, 0]
        xs = kps[vis, 1]
        cy = (float(ys.min()) + float(ys.max())) / 2
        cx = (float(xs.min()) + float(xs.max())) / 2
        span = max(float(ys.max()) - float(ys.min()),
                   float(xs.max()) - float(xs.min()),
                   1e-6)
        scale = (CANVAS * 0.80) / span
    else:
        cy, cx, scale = 0.5, 0.5, CANVAS * 0.80

    half = CANVAS / 2

    def px(y_norm: float, x_norm: float) -> tuple[int, int]:
        return (int((x_norm - cx) * scale + half),
                int((y_norm - cy) * scale + half))

    return px


def render_mannequin(keypoints: list) -> bytes:
    """
    Draw a clean 2-D skeleton mannequin on a light-gray canvas.

    Parameters
    ----------
    keypoints : list of [y_norm, x_norm, confidence] for 17 body landmarks.

    Returns
    -------
    bytes  PNG image data.
    """
    kps     = np.array(keypoints, dtype=np.float32)  # (17, 3)
    visible = kps[:, 2] >= CONF_THRESH
    px      = _make_px(kps)

    canvas = np.full((CANVAS, CANVAS, 3), 232, dtype=np.uint8)

    # --- torso fill -------------------------------------------------------
    torso_idx = [L_SH, R_SH, R_HIP, L_HIP]
    if all(visible[i] for i in torso_idx):
        pts = np.array([px(kps[i, 0], kps[i, 1]) for i in torso_idx], np.int32)
        cv2.fillPoly(canvas, [pts], _FILL)

    # --- skeleton lines ---------------------------------------------------
    for (a, b, color) in CONNECTIONS:
        if visible[a] and visible[b]:
            cv2.line(canvas, px(kps[a, 0], kps[a, 1]),
                     px(kps[b, 0], kps[b, 1]), color, 4, cv2.LINE_AA)

    # --- head: filled dark circle on nose, radius ≈ 0.4 × shoulder width --
    if visible[NOSE]:
        nose_pt = px(kps[NOSE, 0], kps[NOSE, 1])
        if visible[L_SH] and visible[R_SH]:
            ls = px(kps[L_SH, 0], kps[L_SH, 1])
            rs = px(kps[R_SH, 0], kps[R_SH, 1])
            radius = max(12, int(abs(rs[0] - ls[0]) * 0.4))
        else:
            radius = 26
        cv2.circle(canvas, nose_pt, radius, _DARK,  -1, cv2.LINE_AA)  # filled
        cv2.circle(canvas, nose_pt, radius, _WHITE,   2, cv2.LINE_AA)  # outline

    # --- joints -----------------------------------------------------------
    for idx in range(5, 17):
        if not visible[idx]:
            continue
        pt    = px(kps[idx, 0], kps[idx, 1])
        color = _BLUE if idx in _LEFT_JOINTS else _RED
        cv2.circle(canvas, pt, 6, color,  -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 6, _WHITE,  1, cv2.LINE_AA)

    # --- encode -----------------------------------------------------------
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("OpenCV failed to encode PNG")
    return buf.tobytes()
