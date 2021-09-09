from typing import Union
import torch
from functools import wraps

from aloscene import Frame


def assert_and_export_onnx():
    def decorator(forward):
        @wraps(forward)
        def wrapper(instance, frame1, frame2, is_export_onnx=False, *args, **kwargs):
            # A little hack: we use is_export_onnx=None as True when exporting onnx
            # because torch.onnx.export accepts only torch.Tensor or None
            if is_export_onnx is None:
                assert isinstance(frame1, torch.Tensor)
                assert isinstance(frame2, torch.Tensor)
                assert (frame1.ndim == 4) and (frame2.ndim == 4)
                return forward(instance, frame1, frame2, is_export_onnx=None, **kwargs)
            else:
                for frame in [frame1, frame2]:
                    assert frame.normalization == "minmax_sym"
                    assert frame.names == ("B", "C", "H", "W")
                frame1 = frame1.as_tensor()
                frame2 = frame2.as_tensor()

                return forward(instance, frame1, frame2, **kwargs)

        return wrapper

    return decorator
