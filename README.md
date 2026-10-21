# Surgical Frame Labeler

A containerized pipeline that extracts frames from a surgical video and labels each one with a zero-shot CLIP vision-language model. It is built as a Docker learning project for the ASU capstone Developing Vision-Language Models for Surgical Video Analysis.

## Architecture

```
 browser ──► ui (Streamlit, :8501) ──HTTP──► inference (FastAPI + CLIP, :8000)
                     │                                  │
                     └──── reads ──► ./data ◄── writes ──┘
                                (shared volume)
```

| Service | What it does |
|---|---|
| `inference` | Receives a video, samples frames with OpenCV, and scores each frame against text labels with CLIP. Saves the frames and `results.json` to `/data/output/<job_id>/`. |
| `ui` | Lets you upload a clip and edit labels. Calls the inference API and shows the labeled frames and a label-count chart. |

Docker features used: two custom Dockerfiles, layer caching (dependencies installed before code), model weights baked in at build time, a Compose network with service-name DNS, a health check with `depends_on: service_healthy`, and a shared bind-mount volume (read-only on the UI side).

## Run it

Requires Docker Desktop.

```bash
docker compose up --build
```

The first build downloads PyTorch and the CLIP weights, which takes a few minutes. Then open http://localhost:8501, upload a clip, and click **Analyze**.

Self-check (runs a synthetic video through the API):

```bash
docker compose run --rm inference python test_app.py
```

Useful commands for screenshots:

```bash
docker ps                      # both containers running
docker images                  # image sizes
docker compose logs inference  # request logs
```

## Test videos

Use any short, publicly available laparoscopic surgery clip (for example a 30–60 second MP4). The project does not include any patient data.

## Notes

- CLIP was not trained on surgical images, so zero-shot accuracy is limited. The point of the project is a reproducible pipeline that a fine-tuned surgical VLM could later drop into.
- It runs on CPU. For GPU use, switch to a CUDA PyTorch image and add `gpus: all` with the NVIDIA Container Toolkit.
