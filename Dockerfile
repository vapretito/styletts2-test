FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "--login", "-c"]

# Dependencias básicas
RUN apt-get update && apt-get install -y git curl ffmpeg sox unzip build-essential libsox-dev

# Miniforge
RUN curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniforge3-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:$PATH"

RUN conda create -n styletts python=3.10 -y
ENV PATH="/opt/conda/envs/styletts/bin:$PATH"
ENV CONDA_DEFAULT_ENV=styletts

# ⬇️ Copiar todo el contenido actual
WORKDIR /workspace
COPY . /workspace

# ✅ Agregar PYTHONPATH para que reconozca StyleTTS
ENV PYTHONPATH="/workspace/StyleTTS2:$PYTHONPATH"

# Instalar dependencias
WORKDIR /workspace/StyleTTS2
RUN pip install -r requirements.txt
RUN pip install runpod torchaudio requests

# ⬅️ Ejecutar handler correcto
CMD ["python", "/workspace/runpod_handler_styletts2_auto.py"]
