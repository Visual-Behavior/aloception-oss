
# Deformable Detr

## Install

To work properly deformable-detr required custom cuda ops to be built. To build the Multi-scale Deformable Attention ops:

```
cd alonet/deformable_detr/ops
./make.sh
python test.py # should yield True
```

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
