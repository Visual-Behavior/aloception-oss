# tagged aloception-oss:cuda-11.3.1-pytorch1.10.1-lightning1.4.1
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y build-essential nano git wget libgl1-mesa-glx
# Usefull for scipy 
RUN apt-get install -y gfortran 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
ENV PATH=$PATH:/miniconda/condabin:/miniconda/bin
RUN /bin/bash -c "source activate base"
ENV HOME /workspace
WORKDIR /workspace
RUN conda install python=3.9 pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 opencv=4.5.3 -c pytorch -c conda-forge
COPY requirements.txt /install/requirements.txt
RUN pip install -r /install/requirements.txt
