import io
import cv2
import numpy as np

CANVAS = 512
PAD = 56          # pixels of whitespace around the figure
CONF_THRESH = 0.2

# MoveNet keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16

# BGR colors
_DARK = (45, 52, 54)          # near-black — head, torso spine
_BLUE = (210, 140, 55)        # left side  (BGR → appears blue)
_RED  = (55,  75, 220)        # right side (BGR → appears red)
_FILL = (215, 210, 200)       # torso polygon fill

# (joint_a, joint_b, color)
CONNECTIONS = [
    # face
    (NOSE,  L_EYE, _DARK), (NOSE,  R_EYE, _DARK),
    (L_EYE, L_EAR, _DARK), (R_EYE, R_EAR, _DARK),
    # torso
    (L_SH,  R_SH,  _DARK), (L_SH,  L_HIP, _DARK),
    (R_SH,  R_HIP, _DARK), (L_HIP, R_HIP, _DARK),
    # left limbs
    (L_SH, L_EL, _BLUE), (L_EL, L_WR, _BLUE),
    (L_HIP, L_KN, _BLUE), (L_KN, L_AN, _BLUE),
    # right limbs
    (R_SH, R_EL, _RED), (R_EL, R_WR, _RED),
    (R_HIP, R_KN, _RED), (R_KN, R_AN, _RED),
]

_LEFT_JOINTS  = {L_SH, L_EL, L_WR, L_HIP, L_KN, L_AN}
_RIGHT_JOINTS = {R_SH, R_EL, R_WR, R_HIP, R_KN, R_AN}


def _px(y_norm: float, x_norm: float) -> tuple[int, int]:
    """Map a normalized [0,1] keypoint to canvas pixel (x, y)."""
    span = CANVAS - 2 * PAD
    return (int(x_norm * span + PAD), int(y_norm * span + PAD))


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
    kps = np.array(keypoints, dtype=np.float32)  # (17, 3)
    visible = kps[:, 2] >= CONF_THRESH

    canvas = np.full((CANVAS, CANVAS, 3), 232, dtype=np.uint8)

    # --- torso fill --------------------------------------------------------
    torso_idx = [L_SH, R_SH, R_HIP, L_HIP]
    if all(visible[i] for i in torso_idx):
        pts = np.array([_px(kps[i, 0], kps[i, 1]) for i in torso_idx], np.int32)
        cv2.fillPoly(canvas, [pts], _FILL)

    # --- skeleton lines ----------------------------------------------------
    for (a, b, color) in CONNECTIONS:
        if visible[a] and visible[b]:
            cv2.line(canvas, _px(kps[a, 0], kps[a, 1]),
                     _px(kps[b, 0], kps[b, 1]), color, 4, cv2.LINE_AA)

    # --- head circle -------------------------------------------------------
    head_candidates = [i for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if visible[i]]
    if head_candidates:
        ys = [kps[i, 0] for i in head_candidates]
        xs = [kps[i, 1] for i in head_candidates]
        cy, cx = float(np.mean(ys)), float(np.mean(xs))

        if visible[L_SH] and visible[R_SH]:
            ls, rs = _px(kps[L_SH, 0], kps[L_SH, 1]), _px(kps[R_SH, 0], kps[R_SH, 1])
            radius = max(14, int(abs(rs[0] - ls[0]) * 0.25))
        else:
            radius = 22

        cv2.circle(canvas, _px(cy, cx), radius, _DARK, 3, cv2.LINE_AA)

    # --- joints ------------------------------------------------------------
    for idx in range(5, 17):          # skip face keypoints
        if not visible[idx]:
            continue
        pt = _px(kps[idx, 0], kps[idx, 1])
        color = _BLUE if idx in _LEFT_JOINTS else _RED
        cv2.circle(canvas, pt, 6, color,            -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 6, (255, 255, 255),   1, cv2.LINE_AA)

    # --- encode PNG --------------------------------------------------------
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("OpenCV failed to encode PNG")
    return buf.tobytes()
