<p align="center">
  <img src="images/aloception.png" style="text-align:center; width: 50%;" alt="Logo aloception" />
</p>

<a href="https://visual-behavior.github.io/aloception/">Documenation</a> 

# About Aloception

**Aloception** is a set of packages for computer vision built on top of popular deep learning libraries:
[pytorch](<https://pytorch.org/>)  and  [pytorch lightnig](https://www.pytorchlightning.ai/).


**Aloscene** extend the use of
[tensors](https://pytorch.org/tutorials/beginner/examples_tensor/two_layer_net_tensor.html) with **Augmented Tensors** designed to facilitate the use of computer vision data
(such as frames, 2d boxes, 3d boxes, optical flow, disparity, camera parameters...).  


```python
frame = aloscene.Frame("/path/to/image.jpg")
frame = frame.to("cpu")
frame.get_view().render()
```

**Alodataset** implement ready-to-use datasets for computer vision with the help of **aloscene** and **augmented tensors** to make it easier to transform and display your vision data.

```python
coco_dataset = alodataset.CocoDetectionDataset(sample=True)
for frame in coco_dataset.stream_loader():
    frame.get_view().render()
```

**Alonet** integrates several promising computer vision architectures. You can use it for research purposes or to finetune and deploy your model using TensorRT. Alonet is mainly built on top  of [ lightnig](https://www.pytorchlightning.ai/) with the help of
  **aloscene** and **alodataset**.

```python
# Load model
model = alonet.detr.DetrR50(num_classes=91, weights="detr-r50").eval()

# Open and normalized frame
frame = aloscene.Frame("/path/to/image.jpg").norm_resnet()

# Run inference
pred_boxes = model.inference(model([frame]))

# Add and display the predicted boxes
frame.append_boxes2d(pred_boxes[0], "pred_boxes")
frame.get_view().render()
```


### Note
One can use **aloscene** independently than the two other packages to handle computer vision data, or to improve its
training pipelines with **augmented tensors**.

## Install

Aloception's packages are built on top of multiple libraries. Most of them are listed in the **requirements.txt**
```
pip install -r requirements.txt
```

Once the others packages are installed, you still need to install pytorch based on your hardware and environment
configuration. Please, ref to the `pytorch website <https://pytorch.org/>`_  for this install.

## Getting started

<ul>
  <li><a href="https://visual-behavior.github.io/aloception/getting_started/getting_started.html">Getting started</a></li>
  <li><a href="https://visual-behavior.github.io/aloception/getting_started/aloscene.html">Aloscene: Computer vision with ease</a></li>
  <li><a href="https://visual-behavior.github.io/aloception/getting_started/alodataset.html">Alodataset: Loading your vision datasets</a></li>
  <li><a href="https://visual-behavior.github.io/aloception/getting_started/alonet.html">Alonet: Loading & training your models</a></li>
  <li><a href="https://visual-behavior.github.io/aloception/getting_started/augmented_tensor.html">About augmented tensors</a></li>
</ul>


## Tutorials

<ul>
  <li><a href="https://visual-behavior.github.io/aloception/tutorials/data_setup.html">How to setup your data?</a></li>
  <li><a href="https://visual-behavior.github.io/aloception/tutorials/training_detr.html">Training Detr</a></li>
  <li><a href="https://visual-behavior.github.io/aloception/tutorials/finetuning_detr.html">Finetuning DETR</a></li>
  <li><a href="https://visual-behavior.github.io/aloception/tutorials/training_deformable_detr.html">Training Deformable DETR</a></li>
  <li><a href="https://visual-behavior.github.io/aloception/tutorials/finetuning_deformable_detr.html">Finetuning Deformanble DETR</a></li>
  <li><a href="https://visual-behavior.github.io/aloception/tutorials/tensort_inference.html">Exporting DETR / Deformable-DETR to TensorRT</a></li>
</ul>

# Alonet

## Models

| Model name  | Link    | alonet location  | Learn more
|---|---|---|---|
| detr-r50  | https://arxiv.org/abs/2005.12872   | alonet.detr.DetrR50 | <a href="#detr">Detr</a>
| deformable-detr  | https://arxiv.org/abs/2010.04159  | alonet.deformable_detr.DeformableDETR  | <a href="#deformable-detr">Deformable detr</a>
| RAFT | https://arxiv.org/abs/2003.12039 | alonet.raft.RAFT  | <a href="#raft">  RAFT </a> |   |


## Detr

Here is a simple example to get started with **Detr** and aloception. To learn more about Detr, you can checkout the <a href="#tutorials">Tutorials<a/> or the <a href="./alonet/detr">detr README</a>.

```python
# Load model
model = alonet.detr.DetrR50(num_classes=91, weights="detr-r50").eval()

# Open and normalized frame
frame = aloscene.Frame("/path/to/image.jpg").norm_resnet()

# Run inference
pred_boxes = model.inference(model([frame]))

# Add and display the predicted boxes
frame.append_boxes2d(pred_boxes[0], "pred_boxes")
frame.get_view().render()
```
 
## Deformable Detr

Here is a simple example to get started with **Deformable Detr** and aloception. To learn more about Deformable, you can checkout the <a href="#tutorials">Tutorials<a/> or the <a href="./alonet/deformable_detr">deformable detr README</a>.

```python
# Loading Deformable model
model = alonet.deformable_detr.DeformableDetrR50(num_classes=91, weights="deformable-detr-r50").eval()

# Open, normalize frame and send frame on the device
frame = aloscene.Frame("/home/thibault/Desktop/yoga.jpg").norm_resnet().to(torch.device("cuda"))

# Run inference
pred_boxes = model.inference(model([frame]))

# Add and display the predicted boxes
frame.append_boxes2d(pred_boxes[0], "pred_boxes")
frame.get_view().render()
```
  
## RAFT

Here is a simple example to get started with **RAFT** and aloception. To learn more about RAFT, you can checkout the <a href="./alonet/raft">raft README</a>.
  
```python
# Use the left frame from the  Sintel Flow dataset and normalize the frame for the RAFT Model
frame = alodataset.SintelFlowDataset(sample=True).getitem(0)["left"].norm_minmax_sym()

# Load the model using the sintel weights
raft = alonet.raft.RAFT(weights="raft-sintel")

# Compute optical flow
padder = alonet.raft.utils.Padder()
flow = raft.inference(raft(padder.pad(frame[0:1]), padder.pad(frame[1:2])))

# Render the flow along with the first frame
flow[0].get_view().render()
```
  

# Alodataset


## Datasets

| Dataset name  | alodataset location  | To try
|---|---|---|
| CocoDetection  | alodataset.CocoDetectionDataset   | `python alodataset/coco_detection_dataset.py`
| CrowdHuman  | alodataset.CrowdHumanDataset   | `python alodataset/crowd_human_dataset.py `
| Waymo  | alodataset.WaymoDataset   | `python alodataset/waymo_dataset.py`
| ChairsSDHom | alodataset.ChairsSDHomDataset | `python alodataset/chairssdhom_dataset.py`
| FlyingThings3DSubset | alodataset.FlyingThings3DSubsetDataset | `python alodataset/flyingthings_3D_subset_dataset.py`
| FlyingChairs2 | alodataset.FlyingChairs2Dataset | `python alodataset/flying_chairs2_dataset.py`
| Sintel | alodataset.SintelDataset | `python alodataset/sintel_dataset.py`



# Unit tests

```
python -m pytest
```

# Licence

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

