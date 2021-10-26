"""
ModelHandler defines a custom model handler.
"""
import logging
import io
import base64
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

from alonet.deformable_detr.ops.functions import load_ops

logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    image_processing = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    threshold = 0.2
    background_class = 91
    activation_fn = "sigmoid"

    def initialize(self, context):
        load_ops()
        super().initialize(context)
        self.initialized = True
        # TODO : Get hyperparameters from context
        self.threshold = 0.2
        self.background_class = 91
        self.activation_fn = "sigmoid"

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
        outs_boxes, outs_logits = super().inference(data, *args, **kwargs)
        print(outs_logits.shape, outs_boxes.shape)

        # Get probs and labels
        if self.activation_fn == "softmax":
            outs_probs = F.softmax(outs_logits, -1)
        else:
            outs_probs = outs_logits.sigmoid()
        outs_scores, outs_labels = outs_probs.max(-1)

        # Filter by score and threshold
        filters = []
        for scores, labels in zip(outs_scores, outs_labels):
            if self.activation_fn == "softmax":
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
        return [{k: v.tolist() for k, v in batch.items()} for batch in data]


if __name__ == "__main__":
    device = torch.device("cuda")
    load_ops()

    server = ModelHandler()
    server.model = torch.jit.load("deformable-detr-r50.pt")
    server.device = device

    example_input = torch.rand(2, 4, 800, 1333).to(device)
    m_outputs = server.postprocess(server.inference(example_input))
    print(m_outputs)
