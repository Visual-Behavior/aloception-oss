ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04

# Stage 1: Install dependencies
FROM ${BASE_IMAGE} as base
ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update; apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    wget \
    libjpeg-dev \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*
ENV PATH /opt/conda/bin:$PATH

# Stage 2: Install conda
FROM base as conda
ARG PYTHON_VERSION=3.10
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV CONDA_HOME /opt/conda
ENV PATH ${CONDA_HOME}/condabin:${CONDA_HOME}/bin:${PATH}
RUN /opt/conda/bin/conda install -y python=${PYTHON_VERSION}
RUN /opt/conda/bin/conda clean -ya

# Stage 3: Install pytorch
FROM conda as pytorch
ARG PYTORCH_VERSION=2.7.0
ARG TORCHVISION_VERSION=0.22.0
ARG TORCHAUDIO_VERSION=2.7.0
COPY --from=conda /opt/conda /opt/conda
# Install pytorch
RUN /opt/conda/bin/pip install --no-cache-dir torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu126
# Install requirement
COPY requirements/requirements.txt /home/aloception/install/requirements.txt
RUN /opt/conda/bin/pip install --no-cache-dir -r /home/aloception/install/requirements.txt && /opt/conda/bin/conda clean -ya

# Stage 4: Official image
FROM ${BASE_IMAGE} as official
ARG PYTORCH_VERSION=2.7.0
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    sudo \
    build-essential \
    nano \
    git \
    wget \
    libgl1-mesa-glx \
    gfortran \
    libglib2.0-0 \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=pytorch /opt/conda /opt/conda

ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV PYTORCH_VERSION ${PYTORCH_VERSION}

# Create aloception user
RUN useradd --create-home --uid 1000 --shell /bin/bash aloception && usermod -aG sudo aloception && echo "aloception ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
ENV HOME /home/aloception
WORKDIR /home/aloception
USER aloception

# The following so that any user can install packages inside this Image
# RUN chmod -R o+w /opt/conda && chmod -R o+w /home/aloception
COPY --chown=aloception:aloception  ./aloscene/utils /home/aloception/install/utils
USER root
COPY entrypoint.sh  /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
