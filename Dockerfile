FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive
ARG HOME=/home/aloception

RUN apt-get -y  update; apt-get -y install sudo
RUN apt-get install -y build-essential nano git wget libgl1-mesa-glx gfortran libglib2.0-0

# Create aloception user
RUN useradd --create-home --uid 1000 --shell /bin/bash aloception && usermod -aG sudo aloception && echo "aloception ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
ENV HOME /home/aloception
WORKDIR /home/aloception
USER aloception

COPY --chown=aloception:aloception requirements/requirements-torch2.6.0.txt /home/aloception/install/requirements-torch2.6.0.txt
RUN pip install -r /home/aloception/install/requirements-torch2.6.0.txt
COPY --chown=aloception:aloception  ./aloscene/utils /home/aloception/install/utils

USER root
COPY entrypoint.sh  /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
