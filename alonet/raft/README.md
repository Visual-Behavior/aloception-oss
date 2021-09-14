# Raft


Here is a simple example to get started with **RAFT** and aloception. To learn more about RAFT, you can checkout the script bellow.
  
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
  
## Running inference with RAFT

Inference can be performed using official pretrained weights:
```
python alonet/raft/raft.py --weights=raft-things image1.png img2.png
```
or with a custom weight file:
```
python alonet/raft/raft.py --weights=path/to/weights.pth image1.png img2.png
```

## Running evaluation of RAFT
RAFT model is evaluated with official weights "raft-things".

The **EPE** score is computed on Sintel training set. This reproduces the results from RAFT paper : table 1 (training data:"C+T", "method:our(2-views), eval:"sintel training clean).

```
python alonet/raft/eval_on_sintel.py
```

This results in EPE=1.46, which is similar to 1.43 in the paper (obtained as a median score for models trained with 3 differents seeds).

## Training RAFT from scratch
To reproduce the first stage of RAFT training on FlyingChairs dataset:
```
python alonet/raft/train_on_chairs.py
```

It is possible to reproduce the full RAFT training or to train on custom dataset by creating the necessary dataset classes and following the same approach as in `train_on_chairs.py`
