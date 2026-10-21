"""Web UI: upload a surgical video, send it to the inference container,
and show the labeled frames it wrote to the shared /data volume."""

import os
from collections import Counter
from pathlib import Path

import requests
import streamlit as st

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8000")
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))

st.set_page_config(page_title="Surgical Frame Labeler", layout="wide")
st.title("Surgical Frame Labeler")
st.caption("Zero-shot surgical tool recognition with CLIP, running in Docker containers.")

with st.sidebar:
    st.header("Settings")
    labels = st.text_area(
        "Labels (one per line)",
        "a surgical grasper\nsurgical scissors\na surgical hook\na clip applier\nno surgical tool visible",
        height=150,
    )
    every_sec = st.slider("Sample one frame every N seconds", 0.5, 10.0, 2.0, 0.5)
    max_frames = st.slider("Max frames", 4, 60, 24, 4)
    try:
        health = requests.get(f"{INFERENCE_URL}/health", timeout=3).json()
        st.success(f"Inference service up ({health['model']})")
    except requests.RequestException:
        st.error("Inference service unreachable")

video = st.file_uploader("Upload a surgical video clip", type=["mp4", "avi", "mov", "mkv"])

if video and st.button("Analyze", type="primary"):
    with st.spinner("Extracting frames and running CLIP..."):
        try:
            r = requests.post(
                f"{INFERENCE_URL}/analyze",
                files={"video": (video.name, video.getvalue(), video.type)},
                data={"labels": labels, "every_sec": every_sec, "max_frames": max_frames},
                timeout=600,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    result = r.json()
    frames = result["frames"]
    job_dir = DATA_DIR / "output" / result["job_id"]
    st.subheader(f"Job {result['job_id']}: {len(frames)} frames")

    counts = Counter(f["label"] for f in frames)
    st.bar_chart({label: counts.get(label, 0) for label in result["labels"]})

    cols = st.columns(4)
    for i, f in enumerate(frames):
        with cols[i % 4]:
            st.image(str(job_dir / f["frame"]), use_container_width=True)
            st.markdown(f"**{f['label']}**  \n{f['confidence']:.0%} at {f['time_sec']}s")

    st.download_button("Download results.json", (job_dir / "results.json").read_bytes(),
                       file_name=f"results_{result['job_id']}.json")
