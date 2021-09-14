# Detr

Here is a simple example to get started with **Detr** and aloception. To learn more about Detr, you can checkout the <a href="https://visual-behavior.github.io/aloception/tutorials/training_detr.html">Detr Tutorials<a/> or
the scripts described bellow.

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
