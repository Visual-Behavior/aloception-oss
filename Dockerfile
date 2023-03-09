# tagged aloception-oss:cuda-11.3.1-pytorch1.10.1-lightning1.4.1

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

ARG py=3.9
ARG pytorch=1.13.1
ARG torchvision=0.14.1
ARG torchaudio=0.13.1
ARG pytorch_lightning=1.9.0
ARG pycyda=11.7

ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y build-essential nano git wget libgl1-mesa-glx

# Usefull for scipy 
RUN apt-get install -y gfortran 
# required for aloscene
RUN apt-get install -y libglib2.0-0


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
ENV PATH=$PATH:/miniconda/condabin:/miniconda/bin
RUN /bin/bash -c "source activate base"
ENV HOME /workspace
WORKDIR /workspace

# Pytorch & pytorch litning
RUN conda install pytorch==${pytorch} torchvision==${torchvision} torchaudio==${torchaudio} pytorch-cuda=${pycuda} -c pytorch -c nvidia
RUN pip install pytorch_lightning==${pytorch_lightning}

COPY requirements-torch1.13.1.txt /install/requirements-torch1.13.1.txt
RUN pip install -r /install/requirements-torch1.13.1.txt
