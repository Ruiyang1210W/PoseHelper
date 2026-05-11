import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.model import load_models, predict_pose
from app.renderer import render_mannequin

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_MIME_PREFIXES = ("image/jpeg", "image/png", "image/webp", "image/gif", "image/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(title="PoseHelper", version="1.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("frontend/index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not any(file.content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, or WEBP).")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict_pose(image_bytes)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    try:
        mannequin_png = render_mannequin(result["keypoints"])
    except Exception as exc:
        logger.exception("Renderer failed")
        raise HTTPException(status_code=500, detail=f"Render error: {exc}") from exc

    return JSONResponse({
        "label":           result["label"],
        "confidence":      result["confidence"],
        "all_scores":      result["all_scores"],
        "mannequin_image": base64.b64encode(mannequin_png).decode(),
    })
