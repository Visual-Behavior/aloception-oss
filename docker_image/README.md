# Build
```
sudo docker build . -t aloception/aloception
```

# Run
```
sudo docker run --gpus all --ipc host -it --rm -v /PATH/TO/DATA/:/data/ -v /PATH/TO/ALOCEPTION/:/workspace/aloception aloception/aloception /bin/bash
```
