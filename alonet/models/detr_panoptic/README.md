# Detr Panoptic

Here is a simple example to get started with **PanopticHead** and aloception. To learn more about PanopticHead, you can checkout the <a href="https://visual-behavior.github.io/aloception-oss/tutorials/training_panoptic.html">Detr Tutorials</a> or the scripts described bellow.

```python
# Load model : Panoptic Head needs to load a module based on DETR architecture
detr_model = alonet.detr.DetrR50(num_classes=250)
model = alonet.detr_panoptic.PanopticHead(DETR_module=detr_model, weights="detr-r50-panoptic").eval()

# Or simply, you can use the predefined model
model = alonet.detr_panoptic.DetrR50Panoptic(weights="detr-r50-panoptic").eval()

# Open and normalized frame
frame = aloscene.Frame("/path/to/image.jpg").norm_resnet()

# Run inference
pred_boxes, pred_masks = model.inference(model([frame]))

# Add and display the boxes/masks predicted
frame.append_boxes2d(pred_boxes[0], "pred_boxes")
frame.append_segmentation(pred_masks[0], "pred_masks")
frame.get_view().render()
```

### Running inference with DetrR50 + PanopticHead

```
python alonet/detr_panoptic/detr_r50_panoptic.py /path/to/image.jpg
```

### Training Panoptic Detr from scratch
```
python alonet/detr_panoptic/train_on_coco.py
```

### Running evaluation of detr-r50-panoptic

```
python alonet/detr_panoptic/eval_on_coco.py --weights detr-r50-panoptic [--ap_limit n]
```

```
     		 |  all  |  .50  |  .55  |  .60  |  .65  |  .70  |  .75  |  .80  |  .85  |  .90  |  .95  |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
 box		 | 33.85 | 49.88 | 47.83 | 45.33 | 42.58 | 39.36 | 35.12 | 30.25 | 24.41 | 16.71 |  7.03 |
 mask		 | 24.64 | 41.50 | 39.13 | 36.44 | 33.24 | 29.44 | 24.91 | 19.85 | 13.70 |  6.77 |  1.38 |
 precision	 | 31.75 | 43.25 | 41.90 | 40.31 | 38.43 | 36.21 | 33.20 | 29.70 | 25.30 | 19.04 | 10.14 |
 recall		 | 44.18 | 60.96 | 58.92 | 56.52 | 53.70 | 50.39 | 46.07 | 41.02 | 34.66 | 25.85 | 13.69 |
 box_ct		 | 56728.0 | 56728.0 | 56728.0 | 56728.0 | 56728.0 | 56728.0 | 56728.0 | 56728.0 | 56728.0 | 56728.0 | 56728.0 |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+

-------+-------+-------+-------+-------+-------+
                       |  PQst |  SQst |  RQst |
-------+-------+-------+-------+-------+-------+
-------+-------+-------+-------+-------+-------+
 total = 53            | 0.357 | 0.779 | 0.449 |
-------+-------+-------+-------+-------+-------+

-------+-------+-------+-------+-------+-------+
                       |  PQth |  SQth |  RQth |
-------+-------+-------+-------+-------+-------+
-------+-------+-------+-------+-------+-------+
 total = 80            | 0.480 | 0.795 | 0.595 |
-------+-------+-------+-------+-------+-------+

-------+-------+-------+-------+-------+-------+
                       |    PQ |    SQ |    RQ |
-------+-------+-------+-------+-------+-------+
-------+-------+-------+-------+-------+-------+
 total = 133           | 0.431 | 0.789 | 0.537 |
-------+-------+-------+-------+-------+-------+
```
