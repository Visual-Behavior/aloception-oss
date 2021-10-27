---
title: Serving
created: '2021-09-29T13:55:46.171Z'
modified: '2021-10-26T12:13:08.907Z'
---
​
# Launch Serving
​
```
# Stop any serving if exists
torchserve --stop
rm deformable_detr_r50.mar

# Create model.pt
python alonet/deformable_detr/production/export_to_pt.py

# Generate .mar file
torch-model-archiver --model-name deformable_detr_r50 --version 1.0 --serialized-file deformable-detr-r50.pt --handler alonet/detr/production/model_handler.py --extra-files "alonet/detr/production/index_to_name.json,alonet/deformable_detr/production/setup_config.json"

# Move file into model_store folder
mkdir model_store
mv deformable_detr_r50.mar model_store/

# Launch serving
torchserve --start --model-store model_store --models deformable_detr_r50.mar --ncs
```

```
torchserve --stop
```

# Client request


```
# Verify if models is dispo
curl localhost:8081/models
curl localhost:8081/models/deformable_detr_r50

# Inference
curl localhost:8080/predictions/deformable_detr_r50 -T path/to/image
```
