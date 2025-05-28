from dataclasses import dataclass, field
from alonet.common import BaseConfig


@dataclass
class BaseONNXExporterConfig(BaseConfig):
    save_path: str = field(default=None, metadata={"help": "Path to the onnx file"})
    input_shapes: tuple = field(default=((3, 1280, 1920),), metadata={"help": "Input shapes of the model"})
    input_names: tuple = field(default=None, metadata={"help": "Input names of the model"})
    batch_size: int = field(default=1, metadata={"help": "Batch size of the model"})
    do_constant_folding: bool = field(default=True, metadata={"help": "Do constant folding of the model"})
    verbose: bool = field(default=False, metadata={"help": "Verbose of the model"})
    use_scope_names: bool = field(default=False, metadata={"help": "Use scope names of the model"})
    operator_export_type: str = field(default=None, metadata={"help": "Operator export type of the model"})
    opset_version: int = field(default=13, metadata={"help": "Opset version of the model"})
    ignore_adapt_graph: bool = field(default=False, metadata={"help": "Ignore adapt graph of the model"})


@dataclass
class BaseTRTExporterConfig(BaseConfig):
    onnx_path: str = field(default=None, metadata={"help": "Path to the onnx file"})
    precision: str = field(default="fp32", metadata={"help": "Precision of the model"})
    verbose: bool = field(default=False, metadata={"help": "Verbose of the model"})
    profiling_verbosity: int = field(default=0, metadata={"help": "Profiling verbosity of the model"})
