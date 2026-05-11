# PoseHelper

ML-powered human pose reference tool for artists. Upload a photo, get a predicted yoga pose label and a clean skeleton mannequin you can use as a drawing reference.

## Setup

```bash
pip install -r requirements.txt
```

### Model files

Drop these three files into `models/` before running:

| File | Description |
|---|---|
| `mlp_classifier.pkl` | Trained sklearn MLP classifier |
| `scaler.pkl` | Fitted StandardScaler |
| `classes.json` | Class list, e.g. `["downdog","goddess","plank","tree","warrior2"]` |

MoveNet Thunder is downloaded automatically from TF Hub on first run (~200 MB, cached locally).

## Run

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## API

### `POST /predict`

| Field | Type | Description |
|---|---|---|
| `file` | image upload | JPEG, PNG, or WEBP |

**Response JSON**

```json
{
  "label":           "warrior2",
  "confidence":      0.94,
  "all_scores":      { "downdog": 0.01, "goddess": 0.02, ... },
  "mannequin_image": "<base64 PNG>"
}
```

## Feature vector note

`app/model.py` feeds the classifier a 34-element vector — the `[y, x]` pair for each of MoveNet's 17 keypoints, flattened in keypoint order. If your training CSV used a different column layout (e.g. `x` before `y`, or included confidence scores), adjust the slice in `predict_pose()`:

```python
# current (y, x) order — 34 features
features = keypoints[:, :2].flatten()

# swap to (x, y) order
features = keypoints[:, [1, 0]].flatten()

# include confidence — 51 features
features = keypoints.flatten()
```

## Project layout

```
posehelper/
├── app/
│   ├── main.py       FastAPI app, /predict endpoint
│   ├── model.py      MoveNet inference + MLP classifier
│   └── renderer.py   OpenCV skeleton mannequin
├── frontend/
│   └── index.html    Single-page UI
├── models/           Drop mlp_classifier.pkl, scaler.pkl, classes.json here
├── data/             Drop yoga_keypoints.csv here
└── notebooks/        Training notebooks
```
