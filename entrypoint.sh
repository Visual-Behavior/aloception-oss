#!/bin/bash
set -e

if [ -v LOCAL_USER_ID ]; then
  if id -u ${LOCAL_USER_ID} >/dev/null 2>&1; then
    BASE_USER=aloception
  else
    # Create a new user with the specified UID and GI
    useradd --home /home/aloception --uid $LOCAL_USER_ID --shell /bin/bash dynamic && usermod -aG sudo dynamic && echo "dynamic ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    BASE_USER=dynamic
  fi
else
  BASE_USER=aloception
fi

echo "Starting with UID : $LOCAL_USER_ID, base user: $BASE_USER"

export CONDA_HOME=/opt/miniconda; 
export PATH=${CONDA_HOME}/condabin:${CONDA_HOME}/bin:${PATH}; 
source activate base; 

if [ "$#" -ne 0 ]; then
  su -s /bin/bash $BASE_USER -c "export CONDA_HOME=/opt/miniconda; export PATH=${CONDA_HOME}/condabin:${CONDA_HOME}/bin:${PATH}; source activate base; $@"
else
  su -s /bin/bash $BASE_USER -c "export CONDA_HOME=/opt/miniconda; export PATH=${CONDA_HOME}/condabin:${CONDA_HOME}/bin:${PATH}; source activate base; script -q /dev/null -c 'bash -i'"
fi

