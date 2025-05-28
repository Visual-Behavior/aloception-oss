from typing import Union
import torch
from aloscene import Frame
from functools import wraps


def assert_and_export_onnx(check_mean_std=False, input_mean_std=None):
    def decorator(forward):
        @wraps(forward)
        def wrapper(instance, frames: Union[torch.Tensor, Frame], is_export_onnx=False, *args, **kwargs):
            # A little hack: we use is_export_onnx=None and is_tracing=None as True when exporting onnx
            # because torch.onnx.export accepts only torch.Tensor or None
            if hasattr(instance, "tracing") and instance.tracing:
                assert isinstance(frames, (torch.Tensor, dict))
                if isinstance(frames, torch.Tensor):
                    assert frames.shape[1] == 4  # Image input: expected rgb 3 + mask 1
                else:
                    # Image in DETR format, with fields at least:
                    assert all([x in frames for x in ["dec_outputs", "enc_outputs", "bb_lvl3_mask_outputs"]])
                    assert all([x in frames for x in [f"bb_lvl{i}_src_outputs" for i in range(4)]])
                kwargs["is_tracing"] = None
                if is_export_onnx is None:
                    return forward(instance, frames, is_export_onnx=None, **kwargs)
            else:
                if isinstance(frames, list):
                    frames = Frame.batch_list(frames)
                assert isinstance(frames, Frame)
                assert frames.normalization == "resnet"
                assert frames.names == ("B", "C", "H", "W")
                assert frames.mask is not None
                assert frames.mask.names == ("B", "C", "H", "W")
                if check_mean_std and input_mean_std is not None:
                    assert frames.mean_std[0] == input_mean_std[0]
                    assert frames.mean_std[1] == input_mean_std[1]
            return forward(instance, frames, **kwargs)

        return wrapper

    return decorator
