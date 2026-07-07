# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11.6
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

CMD ["python3","inference-scripts/inference-with-roi-from-files.py","--config","/app/config.yml","--param","/app/nanodet-v1/nanodet_v1.ncnn.param","--bin","/app/nanodet-v1/nanodet_v1.ncnn.bin","--device","cpu"]