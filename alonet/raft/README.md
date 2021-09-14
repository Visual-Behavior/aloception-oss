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
