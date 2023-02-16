
# Deformable Detr

## Install

To work properly deformable-detr required custom cuda ops to be built. To build the Multi-scale Deformable Attention ops:

```
cd alonet/deformable_detr/ops
./make.sh
python test.py # should yield True
```

:warning: If you encounter ```error: parameter packs not expanded with ‘...’``` you may need to downgrade gcc and g++ to version <= 10

## Getting started

Here is a simple example to get started with **Deformable Detr** and aloception. To learn more about Deformable, you can checkout the <a href="https://visual-behavior.github.io/aloception-oss/tutorials/training_deformable_detr.html">Deformable Tutorials<a/>.

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

## Running inference with deformable detr

```
python alonet/deformable_detr/deformable_detr_r50.py /path/to/image.jpg
python alonet/deformable_detr/deformable_detr_r50_refinement.py /path/to/image.jpg
```

## Training Deformable Detr R50 from scratch
```
python alonet/deformable_detr/train_on_coco.py --model_name MODEL_NAME
```
With MODEL_NAME either deformable-detr-r50-refinement or deformable-detr-r50 for with/without box refinement.

## Running evaluation of Deformable Detr R50
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

## Exportation
```bash
python trt_exporter.py --refinement --HW 320 480 --verbose
```
or (for preprocessing included)

```bash
python trt_exporter.py --refinement --HW 320 480 --verbose
```