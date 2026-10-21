FROM python:3.11-slim

WORKDIR /app

# OpenCV needs glib even in its headless build.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies before code, so editing app.py doesn't reinstall packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model weights into the image so containers start without downloading.
ENV MODEL_NAME=openai/clip-vit-base-patch32
RUN python -c "from transformers import CLIPModel, CLIPProcessor as P; import os; m=os.environ['MODEL_NAME']; CLIPModel.from_pretrained(m); P.from_pretrained(m)"

COPY app.py test_app.py ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
