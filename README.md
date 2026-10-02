# Digikala Persian Sentiment Analysis — Docker

Dockerized environment for the Persian Digikala sentiment-analysis project.

## Build
docker compose build

## Start
docker compose up

Open http://localhost:8888

## Test NVIDIA GPU
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

Inside the notebook:
```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
```

## Project structure
Put your existing notebook in `notebooks/`.
The large Digikala dataset is intentionally excluded from the ZIP and Git.

## Git
git init
git add .
git commit -m "Dockerize Digikala sentiment analysis environment"

For later changes:
git status
git diff
git add .
git commit -m "Describe the change"

## Research note
For the main end-to-end experiment, keep ParsBERT trainable. Precomputed embeddings are appropriate for a separate frozen-encoder experiment, not an equivalent replacement.
