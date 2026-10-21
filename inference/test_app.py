"""Self-check, run inside the container:
    docker compose run --rm inference python test_app.py
Builds a synthetic 4-second video, sends it to /analyze, and checks the response."""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402

video = Path(tempfile.mkdtemp()) / "test.mp4"
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (224, 224))
for i in range(40):
    writer.write(np.full((224, 224, 3), i * 6, dtype=np.uint8))
writer.release()

client = TestClient(app)
assert client.get("/health").json()["status"] == "ok"

with open(video, "rb") as f:
    r = client.post("/analyze", files={"video": ("test.mp4", f, "video/mp4")},
                    data={"labels": "a red image, a dark image", "every_sec": "1"})
assert r.status_code == 200, r.text
body = r.json()
assert len(body["frames"]) == 4, len(body["frames"])
assert body["labels"] == ["a red image", "a dark image"]
assert all(abs(sum(f["scores"].values()) - 1) < 0.01 for f in body["frames"])
print("All checks passed:", len(body["frames"]), "frames labeled")
