# tagged aloception-oss:cuda-11.3.1-pytorch1.10.1-lightning1.4.1

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

ARG py=3.9
ARG pytorch=2.1.0.dev20230313+cu117
ARG torchvision=0.15.0.dev20230313+cu117
ARG torchaudio=2.0.0.dev20230313+cu117
ARG pytorch_lightning=1.9.3
ARG pycyda=11.7
ARG HOME=/home/aloception

ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y  update; apt-get -y install sudo

RUN apt-get install -y build-essential nano git wget libgl1-mesa-glx

# Usefull for scipy / required for aloscene
RUN apt-get install -y gfortran  libglib2.0-0

# Install gosu
#RUN apt-get install -y gosu && \
#    rm -rf /var/lib/apt/lists/*

# Create aloception user
RUN useradd --create-home --uid 1000 --shell /bin/bash aloception && usermod -aG sudo aloception && echo "aloception ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV HOME /home/aloception
WORKDIR /home/aloception


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh
ENV CONDA_HOME /opt/miniconda
ENV PATH ${CONDA_HOME}/condabin:${CONDA_HOME}/bin:${PATH}
RUN /bin/bash -c "source activate base"
# The following so that any user can install packages inside this Image
#RUN chown -R aloception:aloception /opt/miniconda
RUN chmod -R o+w /opt/miniconda && chmod -R o+w /home/aloception

USER aloception

# Pytorch & pytorch litning
RUN conda install py pytorch-cuda=${pycuda} -c pytorch -c nvidia
RUN pip install --pre torch==${pytorch} torchvision==${torchvision} torchaudio==${torchaudio} --index-url https://download.pytorch.org/whl/nightly/cu117
RUN pip install pytorch_lightning==${pytorch_lightning}

COPY --chown=aloception:aloception requirements/requirements-torch2.1.txt /home/aloception/install/requirements-torch2.1.txt
RUN pip install -r /home/aloception/install/requirements-torch2.1.txt
COPY --chown=aloception:aloception  ./aloscene/utils /home/aloception/install/utils

#USER root
#RUN chmod -R o+w /opt/miniconda && chmod -R o+w /home/aloception
USER root

# Copy the entrypoint script
COPY entrypoint.sh  /entrypoint.sh

#USER root
# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
#RUN cd /home/aloception/install/utils/rotated_iou/cuda_op/; python setup.py install --user