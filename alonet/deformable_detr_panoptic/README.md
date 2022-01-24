# Deformable Detr Panoptic

Here is a simple example to get started with **PanopticHead** and aloception. To learn more about PanopticHead, you can checkout the <a href="https://visual-behavior.github.io/aloception/tutorials/training_panoptic.html">Detr Tutorials</a> or the scripts described bellow.

```python
# Load model : Panoptic Head needs to load a module based on DETR architecture
base_model = alonet.deformable_detr.DeformableDetrR50(num_classes=250)
model = alonet.detr_panoptic.PanopticHead(DETR_module=base_model, weights="deformable-detr-r50-panoptic").eval()

# Or simply, you can use one of the predefined model
model = alonet.deformable_detr_panoptic.DeformableDetrR50Panoptic(weights="deformable-detr-r50-panoptic").eval()
model = alonet.deformable_detr_panoptic.DeformableDetrR50Panoptic(
    with_box_refine=True,
    weights="deformable-detr-r50-refinement-panoptic",
    activation_fn="softmax,
).eval()

# Open and normalized frame
frame = aloscene.Frame("/path/to/image.jpg").norm_resnet()

# Run inference
pred_boxes, pred_masks = model.inference(model([frame]))

# Add and display the boxes/masks predicted
frame.append_boxes2d(pred_boxes[0], "pred_boxes")
frame.append_segmentation(pred_masks[0], "pred_masks")
frame.get_view().render()
```

### Running inference with DeformableDetrR50 + PanopticHead

```
python alonet/deformable_detr_panoptic/deformable_detr_r50_panoptic.py /path/to/image.jpg
```

### Training Panoptic Deformable Detr from scratch
```
python alonet/deformable_detr_panoptic/train_on_coco.py
```

### Running evaluation of detr-r50-panoptic

```
python alonet/deformable_detr_panoptic/eval_on_coco.py --weights deformable-detr-r50-refinement-panoptic --model_name deformable-detr-r50-refinement-panoptic --activation_fn softmax [--ap_limit n]
```

```
             |  all  |  .50  |  .55  |  .60  |  .65  |  .70  |  .75  |  .80  |  .85  |  .90  |  .95  |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
 box         | 30.53 | 43.76 | 42.15 | 40.41 | 38.44 | 35.54 | 32.35 | 28.29 | 22.94 | 15.67 |  5.78 |
 mask        | 19.56 | 32.18 | 30.25 | 28.16 | 26.07 | 23.56 | 20.39 | 16.59 | 11.39 |  5.79 |  1.25 |
 precision   | 30.79 | 40.74 | 39.63 | 38.44 | 37.08 | 34.77 | 32.51 | 29.51 | 25.61 | 19.63 |  9.95 |
 recall      | 42.08 | 55.56 | 54.11 | 52.41 | 50.53 | 47.74 | 44.62 | 40.52 | 34.99 | 26.69 | 13.66 |
 box_ct      | 56728.0 | 56728.0 | 56728.0 | 56728.0 | 56728.0 | 56728.0 | 56728.0 | 56728.0 | 56728.0 | 56728.0 | 56728.0 |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+

-------+-------+-------+-------+-------+-------+
                       |  PQst |  SQst |  RQst |
-------+-------+-------+-------+-------+-------+
-------+-------+-------+-------+-------+-------+
 total = 53            | 0.242 | 0.712 | 0.294 |
-------+-------+-------+-------+-------+-------+

-------+-------+-------+-------+-------+-------+
                       |  PQth |  SQth |  RQth |
-------+-------+-------+-------+-------+-------+
-------+-------+-------+-------+-------+-------+
 total = 80            | 0.443 | 0.790 | 0.546 |
-------+-------+-------+-------+-------+-------+

-------+-------+-------+-------+-------+-------+
                       |    PQ |    SQ |    RQ |
-------+-------+-------+-------+-------+-------+
-------+-------+-------+-------+-------+-------+
 total = 133           | 0.363 | 0.759 | 0.446 |
-------+-------+-------+-------+-------+-------+
```
