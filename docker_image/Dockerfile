FROM nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04
ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y build-essential nano git wget libgl1-mesa-glx
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
ENV PATH=$PATH:/miniconda/condabin:/miniconda/bin
RUN /bin/bash -c "source activate base"
ENV HOME /workspace
WORKDIR /workspace
RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 opencv=4.5.3 -c pytorch -c conda-forge
RUN pip install scipy
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY alodataset_config.json /workspace/.aloception/
ENV PYTHONPATH "${PYTHONPATH}:/workspace/aloception/"
