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

# Usefull for scipy 
RUN apt-get install -y gfortran 
# required for aloscene
RUN apt-get install -y libglib2.0-0
# Create aloception user
RUN useradd -m aloception && echo "aloception:aloception" | chpasswd && adduser aloception sudo
USER aloception

ENV HOME /home/aloception
WORKDIR /home/aloception

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/aloception/miniconda
ENV PATH=$PATH:/home/aloception/miniconda/condabin:/home/aloception/miniconda/bin
RUN /bin/bash -c "source activate base"

RUN chown aloception:aloception /home/aloception

# Pytorch & pytorch litning
RUN conda install py pytorch-cuda=${pycuda} -c pytorch -c nvidia
RUN pip install --pre torch==${pytorch} torchvision==${torchvision} torchaudio==${torchaudio} --index-url https://download.pytorch.org/whl/nightly/cu117
RUN pip install pytorch_lightning==${pytorch_lightning}

COPY --chown=aloception:aloception requirements/requirements-torch2.1.txt /home/aloception/install/requirements-torch2.1.txt
RUN pip install -r /home/aloception/install/requirements-torch2.1.txt
COPY --chown=aloception:aloception  ./aloscene/utils /home/aloception/install/utils

RUN cd /home/aloception/install/utils/rotated_iou/cuda_op/; python setup.py install --user