# CUDA 11.8 runtime — compatible with host driver 575.64 (CUDA 12.9)
# torch 2.0.0 was built against CUDA 11.8 wheels
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps: PortAudio, ALSA, Python
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3 \
        python3-pip \
        python3-dev \
        portaudio19-dev \
        libportaudio2 \
        libasound2-dev \
        libusb-1.0-0-dev \
        alsa-utils \
        ffmpeg \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 11.8 wheels first (separate index URL)
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 \
    torch==2.0.0 \
    torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install remaining runtime deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 \
    numpy==1.21.5 \
    matplotlib \
    pyargus==1.1.post1 \
    pyaudio==0.2.13 \
    Requests==2.31.0

# Copy application source (data/ and notebooks/ excluded via .dockerignore)
COPY . .

# Directory for recorded WAV clips (mounted as a volume at runtime)
RUN mkdir -p /app/clips

EXPOSE 8000

CMD ["python3", "main.py"]
