"""
Defines a custom model handler for models based on :mod:`Detr <alonet.detr.detr>` and
:mod:`Deformable detr <alonet.deformable_detr.deformable_detr>`. For more information, see
`TorchServe <https://pytorch.org/serve/>`_ documentation
"""
import sys
import os
import json
import logging
import io
import base64
from PIL import Image
from argparse import Namespace

import torch
import torch.nn.functional as F
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

from alonet.deformable_detr.ops.functions import load_ops

logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):
    """Define a custom handler for serving on from :mod:`alonet` models

    Attributes
    ----------
    threshold : float
        Threshold value to filter boxes, by default 0.2
    background_class : int
        Background class ID, by default 91
    activation_fn : str
        activation function, either of {``sigmoid``,``softmax``}, use in deformable architectures,
        by default ``softmax``

    Note
    ----
    Use --extra_files path/new/setup_config.json for setup new attributes values on
    `torchmodelarchiver <https://pytorch.org/serve/custom_service.html#creating-a-model-archive-with-an-entry-point>`_
    command.
    """

    def __init__(self):
        super().__init__()
        self.image_processing = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        # Default inference-parameters
        self.threshold = 0.2
        self.background_class = 91
        self.activation_fn = "sigmoid"

    def initialize(self, context):
        # Read additional configuration
        model_dir = context.system_properties.get("model_dir")
        logger.info("test: {}".format(model_dir))
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                setup_config = json.load(setup_config_file)
            self.__dict__.update(setup_config)  # Update parameters
        else:
            logger.warning("Missing the setup_config.json file. Take default values.")

        # Load cuda ops from default path
        load_ops()

        super().initialize(context)

    def preprocess(self, data):
        # Take from ts.torch_handler.vision_handler
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image)).convert("RGB")
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)
            image = torch.cat([image, torch.zeros_like(image)[0:1]], dim=0)  # Append mask
            images.append(image)

        return torch.stack(images).to(self.device)

    def inference(self, data, *args, **kwargs):
        logger.info("context : th={}, af={}, bc={}".format(self.threshold, self.activation_fn, self.background_class))
        outs_boxes, outs_logits = super().inference(data, *args, **kwargs)

        # Get probs and labels
        if self.activation_fn == "softmax":
            outs_probs = F.softmax(outs_logits, -1)
        else:
            outs_probs = outs_logits.sigmoid()
        outs_scores, outs_labels = outs_probs.max(-1)

        # Filter by score and threshold
        filters = []
        for scores, labels in zip(outs_scores, outs_labels):
            if self.activation_fn == "softmax" and self.background_class is not None:
                filters.append((labels != self.background_class) & (scores > self.threshold))
            else:
                filters.append(scores > self.threshold)

        # Make labels and boxes
        pred_boxes = list()

        for scores, labels, boxes, b_filter in zip(outs_scores, outs_labels, outs_boxes, filters):
            boxes = boxes[b_filter]
            labels = labels[b_filter]
            scores = scores[b_filter]
            pred_boxes.append({"boxes": boxes, "labels": labels, "scores": scores})

        return pred_boxes

    def postprocess(self, data):
        def batch_process(batch):
            result = []
            for lbl, box, score in zip(batch["labels"], batch["boxes"], batch["scores"]):
                lbl = str(int(lbl.item()))
                lbl = lbl if self.mapping is None else self.mapping[lbl]
                result.append({lbl: box.tolist(), "score": score.item()})
            return result

        return [batch_process(batch) for batch in data]


if __name__ == "__main__":
    device = torch.device("cuda")

    server = ModelHandler()
    context = Namespace(
        system_properties=dict(gpu_id=0, model_dir="/home/johan/work/aloception"),
        manifest=dict(model=dict(serializedFile="deformable-detr-r50.pt")),
    )
    server.initialize(context)

    if len(sys.argv) > 1:
        data = []
        for i in range(1, len(sys.argv)):
            with open(sys.argv[i], "rb") as fimage:
                example_input = fimage.read()
            data.append({"body": bytearray(example_input)})
        example_input = server.preprocess(data)
    else:
        example_input = torch.rand(2, 4, 800, 1333).to(device)

    m_outputs = server.postprocess(server.inference(example_input))
    print(m_outputs)
