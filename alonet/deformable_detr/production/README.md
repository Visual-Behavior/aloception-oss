---
title: Serving
created: '2021-09-29T13:55:46.171Z'
modified: '2021-09-29T13:57:08.907Z'
---
​
# Launch Serving
​
```
torchserve --stop
```
​
```
rm deformable_detr_r50.mar
```

```
torch-model-archiver --model-name deformable_detr_r50 --version 1.0 --serialized-file deformable-detr-r50.pt --handler alonet/deformable_detr/production/handlers/deformable.py
```
​
```
mkdir model_store
mv deformable_detr_r50.mar model_store/
```
​
```
torchserve --start --model-store model_store --models deformable_detr_r50.mar --ncs
```

```
torchserve --stop
```

# Client request

## Verify if models is dispo

```
curl localhost:8081/models
```

```
curl localhost:8081/models/deformable_detr_r50
```

## Inference

```
curl localhost:8080/predictions/deformable_detr_r50 -T path/to/image
```
