import io
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

logger = logging.getLogger(__name__)

MOVENET_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
INPUT_SIZE = 256  # Thunder uses 256x256
MODELS_DIR = Path("models")

_movenet = None
_classifier = None
_scaler = None
_classes: list[str] = []


def load_models() -> None:
    global _movenet, _classifier, _scaler, _classes

    logger.info("Loading MoveNet Thunder from TF Hub (downloads on first run)…")
    module = hub.load(MOVENET_URL)
    _movenet = module.signatures["serving_default"]
    logger.info("MoveNet ready.")

    with open(MODELS_DIR / "mlp_classifier.pkl", "rb") as f:
        _classifier = pickle.load(f)
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        _scaler = pickle.load(f)

    raw = json.loads((MODELS_DIR / "classes.json").read_text())
    if isinstance(raw, list):
        _classes = raw
    elif isinstance(raw, dict):
        # Support {"0": "downdog", ...} or {"downdog": 0, ...}
        first_val = next(iter(raw.values()))
        if isinstance(first_val, int):
            # name → index mapping: invert it
            _classes = [k for k, _ in sorted(raw.items(), key=lambda x: x[1])]
        else:
            # index → name mapping
            _classes = [raw[str(i)] for i in range(len(raw))]
    else:
        raise ValueError("classes.json must be a list or dict")

    logger.info("Classifier ready. Classes: %s", _classes)


def _run_movenet(image_bytes: bytes) -> np.ndarray:
    """Return (17, 3) array of [y_norm, x_norm, confidence] keypoints."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE))
    tensor = tf.constant(np.array(img, dtype=np.int32)[np.newaxis])
    outputs = _movenet(input=tensor)
    # shape: (1, 1, 17, 3)
    return outputs["output_0"].numpy()[0, 0]


def predict_pose(image_bytes: bytes) -> dict:
    """
    Run MoveNet + MLP classifier on image bytes.

    Feature vector: 51 floats — [y, x, confidence] for each of the 17 MoveNet
    keypoints, flattened in keypoint order (nose … right_ankle).  Matches the
    layout of yoga_keypoints.csv exports that include the confidence column.
    """
    keypoints = _run_movenet(image_bytes)          # (17, 3)
    features = keypoints.flatten()                   # (51,) — [y0,x0,c0, y1,x1,c1, …]
    features_scaled = _scaler.transform(features.reshape(1, -1))

    probs = _classifier.predict_proba(features_scaled)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "label": _classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_scores": {cls: float(p) for cls, p in zip(_classes, probs)},
        "keypoints": keypoints.tolist(),
    }
