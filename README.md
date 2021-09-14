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
frame = alonet.Frame("path/to/frame.jpg")
frame = frame.to("cuda")
frame.get_view().render()
```

**Alodataset** implement ready-to-use datasets for computer vision with the help of **aloscene** and **augmented tensors** to make it easier to transform and display your vision data.

```python
coco_dataset = CocoDetectionDataset(sample=True)
for frames coco_dataset.stream_loader():
    frames.get_view().render()
```

**Alonet** integrates several promising computer vision architectures. You can use it for research purposes or to finetune and deploy your model using TensorRT. Alonet is mainly built on top  of [ lightnig](https://www.pytorchlightning.ai/) with the help of
  **aloscene** and **alodataset**.

```python
# Load model
model = DetrR50(num_classes=91, weights="detr-r50").eval()

# Open and normalized frame
frame = aloscene.Frame("path/to/image").norm_resnet()
frames = aloscene.Frame.batch_list([frame])

# Run inference
pred_boxes = model.inference( model(frames))

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

# Alonet

## Models

| Model name  | Link    | alonet location  | Learn more
|---|---|---|---|
| detr-r50  | https://arxiv.org/abs/2005.12872   | alonet.detr.DetrR50 | <a href="#detr">Detr</a>
| deformable-detr  | https://arxiv.org/abs/2010.04159  | alonet.deformable_detr.DeformableDETR  | <a href="#deformable-detr">Deformable detr</a>
| RAFT | https://arxiv.org/abs/2003.12039 | alonet.raft.RAFT  | <a href="#raft">  RAFT </a> |   |


## Detr

### Running inference with detr_r50

```
python alonet/detr/detr_r50.py /path/to/image.jpg
```

### Training Detr from scratch
```
python alonet/detr/train_on_coco.py
```

### Running evaluation of detr-r50

```
python alonet/detr/eval_on_coco.py --weights detr-r50 --batch_size 1
```

```
     		 |  all  |  .50  |  .55  |  .60  |  .65  |  .70  |  .75  |  .80  |  .85  |  .90  |  .95  |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
 box		 | 39.93 | 58.80 | 56.57 | 54.00 | 50.99 | 47.06 | 42.17 | 36.12 | 28.74 | 18.62 |  6.25 |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
```

## Deformable Detr

### Running inference with deformable detr

```
python alonet/deformable_detr/deformable_detr_r50.py /path/to/image.jpg
python alonet/deformable_detr/deformable_detr_r50_refinement.py /path/to/image.jpg
```

### Training Deformable Detr R50 from scratch
```
python alonet/deformable_detr/train_on_coco.py --model_name MODEL_NAME
```
With MODEL_NAME either deformable-detr-r50-refinement or deformable-detr-r50 for with/without box refinement.

### Running evaluation of Deformable Detr R50
```
python alonet/deformable_detr/eval_on_coco.py --model_name MODEL_NAME --weights MODEL_NAME --batch_size 1 [--ap_limit NUMBER_OF_SAMPLES]
```
With MODEL_NAME either deformable-detr-r50-refinement or deformable-detr-r50 for with/without box refinement.

Evaluation on 1000 images COCO with box refinement
```
                 |  all  |  .50  |  .55  |  .60  |  .65  |  .70  |  .75  |  .80  |  .85  |  .90  |  .95  |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
 box             | 44.93 | 62.24 | 60.26 | 58.00 | 55.76 | 52.51 | 48.07 | 42.99 | 36.13 | 24.28 |  9.02 |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
```

## RAFT

### Running inference with RAFT

Inference can be performed using official pretrained weights:
```
python alonet/raft/raft.py --weights=raft-things image1.png img2.png
```
or with a custom weight file:
```
python alonet/raft/raft.py --weights=path/to/weights.pth image1.png img2.png
```
### Running evaluation of RAFT
RAFT model is evaluated with official weights "raft-things".

The **EPE** score is computed on Sintel training set. This reproduces the results from RAFT paper : table 1 (training data:"C+T", "method:our(2-views), eval:"sintel training clean).

```
python alonet/raft/eval_on_sintel.py
```

This results in EPE=1.46, which is similar to 1.43 in the paper (obtained as a median score for models trained with 3 differents seeds).

### Training RAFT from scratch
To reproduce the first stage of RAFT training on FlyingChairs dataset:
```
python alonet/raft/train_on_chairs.py
```

It is possible to reproduce the full RAFT training or to train on custom dataset by creating the necessary dataset classes and following the same approach as in `train_on_chairs.py`

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

