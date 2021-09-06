# Aloception

<img src="/images/aloception_logo.png" width="100%" alt="Logo aloception" />

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

## About The Project
Aloception is a tool that that facilitate the train process for many architectures based on Transformers. It is flexible, since it allows coupling to different databases and training with them multiple architectures for object detection.

For this, aloception provides different modules that facilitate its use. Each module is developed under the [pytorch-lightning framework](https://pytorch-lightning.readthedocs.io/en/latest/#).

### Built With
* [python](https://rasa.com/)
* [opencv](https://opencv.org/)
* [pytorch](https://pytorch.org/)
* [pytorchlightning](https://www.pytorchlightning.ai/)

## Getting Started
See all [aloception documentation](/README.md) and installation instructions.

## Usage
Each aloception module that is based on [pytorch-lightning framework](https://pytorch-lightning.readthedocs.io/en/latest/#) has five levels of implementation.

```python
from argparse import ArgumentParser

from alonet.common import add_argparse_args
from alonet.detr import DetrR50Finetune
from alonet.detr import CocoDetection2Detr, LitDetr

from alodataset import Mot17, Split

# Level 1
# Ligthning module instace with default parameters
coco_loader = CocoDetection2Detr()

# Level 2
# Using Namespace to define the parameter values (modifiable by console line)
parser = ArgumentParser()
parser = CocoDetection2Detr.add_argparse_args(parser)
parser = LitDetr.add_argparse_args(parser)
parser = add_argparse_args(args) # Common parameters
args = parser.args()

coco_loader = CocoDetection2Detr(args)
lit_detr = LitDetr(args)

# Level 3
# Custom parameters in-line
my_model = DetrR50Finetune(num_classes=2, background_class=0, weights="detr-r50")
lit_detr = LitDetr(model=my_model)

# Level 4
# Combine previous approaches.
lit_detr = LitDetr(args, gradient_clip_val=0,accumulate_grad_batches=2)

# Level 5
# Subclassing for aloception and compatible with lighting modules
# https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

class Mot17DetectionDetr(CocoDetection2Detr):

    # Method overwrite
    def train_dataloader(self):
        return Mot17( # Change default dataset to MOT17
            split=Split.TRAIN,
            sequence_size=1,
            mot_sequences=["MOT17-02-DPM", "MOT17-02-SDP"],
            transform_fn = self.train_transform
        ).train_loader(batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self, limit=200, sampler=None):
        limit = limit if self.limit is None else self.limit
        return Mot17( # Change default dataset to MOT17
            split=Split.TRAIN,
            sequence_size=1,
            mot_sequences=["MOT17-05-DPM", "MOT17-05-SDP"],
            transform_fn = self.val_transform
        ).train_loader(batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)

mot_loader = Mot17DetectionDetr()
```

This allows for different types of training implementations, from training with a minimum of parameters:

```python
from argparse import ArgumentParser

from alonet.common import add_argparse_args
from alonet.detr import CocoDetection2Detr, LitDetr

# Build parser
parser = ArgumentParser()
parser = CocoDetection2Detr.add_argparse_args(parser)
parser = LitDetr.add_argparse_args(parser)
parser = add_argparse_args(args) # Common parameters
args = parser.args()

# Modules instence
coco_loader = CocoDetection2Detr()
lit_detr = LitDetr()

# Train process
lit_detr.run_train(
    data_loader=coco_loader,
    args=args,
    project="detr",
    expe_name="coco_detr",
)

# Make a random prediction
frame = next(iter(coco_loader.val_dataloader()))
frame = frame[0].batch_list(frame).to(device)
pred_boxes = lit_detr.inference(lit_detr(frame))[0] # Inference from forward result
gt_boxes = frame[0].boxes2d

frame.get_view([
    gt_boxes.get_view(frame[0], title="Ground truth boxes"),
    pred_boxes.get_view(frame[0], title="Predicted boxes"),
]).render()
```

Or do some more sophisticated training, in order to make a [finetune training](/tutorials/5.6-custom_detrmod.py).


## Roadmap

See the [open issues](https://gitlab.com/aigen-vision/aloception/issues) for a list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact
*

## Acknowledgements
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template) for providing the README template.
