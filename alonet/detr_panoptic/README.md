# Detr Panoptic

Here is a simple example to get started with **PanopticHead** and aloception. To learn more about PanopticHead, you can checkout the <a href="https://visual-behavior.github.io/aloception/tutorials/training_panoptic.html">Detr Tutorials</a> or
the scripts described bellow.

```python
# Load model : Panoptic Head needs to load a module based on DETR architecture
detr_model = alonet.detr.DetrR50(num_classes=250, background=250)
model = alonet.detr_panoptic.PanopticHead(DETR_module=detr_model, weights="detr-r50-panoptic").eval()

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
python alonet/detr_panoptic/detr_panoptic.py /path/to/image.jpg
```

### Training Panoptic Detr from scratch
```
python alonet/detr_panoptic/train_on_coco.py
```

### Running evaluation of detr-r50-panoptic

```
python alonet/detr_panoptic/eval_on_coco.py --weights detr-r50-panoptic --batch_size 1 [--ap_limit n]
```

```
                 |  all  |  .50  |  .55  |  .60  |  .65  |  .70  |  .75  |  .80  |  .85  |  .90  |  .95  |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
 box             | 31.26 | 45.16 | 43.57 | 41.39 | 39.13 | 36.44 | 32.70 | 28.38 | 23.09 | 15.95 |  6.77 |
 mask            | 24.62 | 41.47 | 39.11 | 36.43 | 33.21 | 29.41 | 24.90 | 19.87 | 13.68 |  6.77 |  1.38 |

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
