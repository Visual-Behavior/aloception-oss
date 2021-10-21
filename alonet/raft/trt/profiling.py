from torch.utils.data import SequentialSampler
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import argparse
import time
import os
import io


from alonet.raft.trt_exporter import load_trt_plugins_raft
from aloscene.tensors import SpatialAugmentedTensor
from alodataset import ChairsSDHomDataset
from alonet.torch2trt import TRTExecutor
from alonet import ALONET_ROOT
from alonet.raft import RAFT


def get_sample_images(for_trt=False):
    dataset = ChairsSDHomDataset(sample=True)
    frame = next(iter(dataset.train_loader(sampler=SequentialSampler)))
    frame = SpatialAugmentedTensor.batch_list(frame)
    frame = frame[:, :, :, 0:368, 0:496]
    frame = frame.norm_minmax_sym()
    frame1 = frame[:, 0, :, :, :]
    frame2 = frame[:, 1, :, :, :]
    if for_trt:
        frame1 = frame1.as_tensor().numpy()
        frame2 = frame2.as_tensor().numpy()
    return frame1, frame2


def load_trt_model(engine_path=None, sync_mode=True, profiling=False):
    if engine_path is None:
        engine_path = os.path.join(ALONET_ROOT, "raft-things_fp32.engine")
    load_trt_plugins_raft()
    model = TRTExecutor(engine_path, sync_mode=sync_mode, profiling=profiling)
    model.print_bindings_info()
    return model


def load_torch_model():
    model = RAFT(weights="raft-things")
    model.eval().to("cuda")
    return model


def profile_trt(engine_path=None, precision="fp32", iters=12, name="raft-things"):

    if engine_path is None:
        engine_path = os.path.join(ALONET_ROOT, f"{name}_iters{iters}_{precision}.engine")

    model = load_trt_model(engine_path, profiling=True)
    frame1, frame2 = get_sample_images(for_trt=True)

    model.inputs[0].host = frame1
    model.inputs[1].host = frame2

    # GPU warm up
    [model.execute() for _ in range(5)]
    print("=====warmup_end=====", flush=True)

    for _ in range(5):
        tic = time.time()
        model.execute()
        toc = time.time()
        print("=====inference_end=====", flush=True)
    print("=====profiling_end=====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="backend")
    torch_parser = subparsers.add_parser("torch")
    trt_parser = subparsers.add_parser("trt")
    trt_parser.add_argument("precision", choices=["fp16", "fp32", "mix"])
    trt_parser.add_argument("--iters", type=int, default=12)
    trt_parser.add_argument("--engine_path")
    trt_parser.add_argument("--name", default="raft-things")
    kwargs = vars(parser.parse_args())
    backend = kwargs.pop("backend")
    if backend == "torch":
        raise NotImplementedError
    else:
        profile_trt(**kwargs)
