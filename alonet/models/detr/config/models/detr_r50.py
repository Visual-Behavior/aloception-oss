from dataclasses import dataclass, field

from alonet.common import BaseConfig


@dataclass
class DetrR50Config(BaseConfig):
    num_classes: int = field(default=91, metadata={"help": "Number of classes in embed layer"})
    background_class: int = field(default=91, metadata={"help": "Background class"})
    aux_loss: bool = field(default=True, metadata={"help": "Use auxiliary loss"})
    weights: str = field(default=None, metadata={"help": "Path to the weights"})
    return_dec_outputs: bool = field(default=False, metadata={"help": "Return decoder outputs"})
    return_enc_outputs: bool = field(default=False, metadata={"help": "Return encoder outputs"})
    return_bb_outputs: bool = field(default=False, metadata={"help": "Return backbone outputs"})
    strict_load_weights: bool = field(default=True, metadata={"help": "Strict load weights"})
    tracing: bool = field(default=False, metadata={"help": "Tracing"})
