"""Inference service: extracts frames from a surgical video and labels them
with a zero-shot CLIP vision-language model."""

import json
import os
import re
import uuid
from pathlib import Path

import cv2
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
BATCH_SIZE = 8
DEFAULT_LABELS = [
    "a surgical grasper",
    "surgical scissors",
    "a surgical hook",
    "a clip applier",
    "no surgical tool visible",
]

app = FastAPI(title="Surgical Frame Labeler")

# Loaded once at startup; the weights were downloaded into the image at build time.
model = CLIPModel.from_pretrained(MODEL_NAME).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)


def extract_frames(video_path: Path, every_sec: float, max_frames: int):
    """Return (timestamp_seconds, PIL.Image) pairs, one every `every_sec` seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not read the video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps * every_sec)), 1)
    frames, idx = [], 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx / fps, Image.fromarray(rgb)))
        idx += 1
    cap.release()
    return frames


@torch.no_grad()
def classify(images, labels):
    """Zero-shot classification: probability of each label for each image."""
    inputs = processor(text=labels, images=images, return_tensors="pt", padding=True)
    return model(**inputs).logits_per_image.softmax(dim=1)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/analyze")
def analyze(
    video: UploadFile = File(...),
    labels: str = Form(""),
    every_sec: float = Form(2.0),
    max_frames: int = Form(24),
):
    label_list = [l.strip() for l in re.split(r"[\n,]", labels) if l.strip()] or DEFAULT_LABELS

    # Each request gets its own folder on the shared /data volume.
    job_id = uuid.uuid4().hex[:8]
    job_dir = DATA_DIR / "output" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / ("input" + (Path(video.filename or "").suffix or ".mp4"))
    video_path.write_bytes(video.file.read())

    frames = extract_frames(video_path, every_sec, max_frames)
    if not frames:
        raise HTTPException(status_code=400, detail="No frames could be extracted.")

    results = []
    for start in range(0, len(frames), BATCH_SIZE):
        chunk = frames[start:start + BATCH_SIZE]
        probs = classify([img for _, img in chunk], label_list)
        for (t, img), p in zip(chunk, probs):
            name = f"frame_{len(results):03d}.jpg"
            img.save(job_dir / name)
            top = int(p.argmax())
            results.append({
                "frame": name,
                "time_sec": round(t, 2),
                "label": label_list[top],
                "confidence": round(float(p[top]), 3),
                "scores": {l: round(float(s), 3) for l, s in zip(label_list, p)},
            })

    summary = {"job_id": job_id, "model": MODEL_NAME, "labels": label_list, "frames": results}
    (job_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary
