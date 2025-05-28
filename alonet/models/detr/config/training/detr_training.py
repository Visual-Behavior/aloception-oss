from dataclasses import dataclass, field

from alonet.common import BaseTrainerConfig


@dataclass
class DetrTrainingConfig(BaseTrainerConfig):
    model_name: str = field(default="detr-r50", metadata={"help": "Name of the model to use"})
    weights: str = field(default=None, metadata={"help": "Path to the weights file"})
    viz_interval: int = field(default=500, metadata={"help": "Interval for logging visualizations"})
