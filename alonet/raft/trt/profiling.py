from torch.utils.data import SequentialSampler
from contextlib import redirect_stdout, redirect_stderr
from collections import defaultdict
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

    model = load_trt_model(engine_path, profiling=True, sync_mode=True)
    frame1, frame2 = get_sample_images(for_trt=True)

    model.inputs[0].host = frame1
    model.inputs[1].host = frame2

    # GPU warm up
    [model.execute() for _ in range(10)]
    print("=====warmup_end=====", flush=True)

    for it in range(5):
        print(f"==== iter #{it} ===")
        model.context.profiler.reset()
        model.execute()
        # model.context.profiler.print_all()
        # plot_timings(model)
        # print_costly_nodes(model)
        # aggregated_time(model)
        # print_op_names(model)
        print_not_in_names(model)
        exit()
        print("=====inference_end=====", flush=True)

    print("=====profiling_end=====")

    # model.context.profiler.print_all()


def plot_timings(model):
    import matplotlib.pyplot as plt

    timings = model.context.profiler.timing
    timings = {key: np.sum(val) for key, val in timings.items()}

    values = sorted(timings.values(), reverse=True)
    total_time = np.sum(values)
    plt.figure()
    plt.subplot(211)
    plt.title("nodes timing in decreasing order")
    plt.plot(values)
    plt.ylabel("time in ms")
    plt.subplot(212)
    plt.title("")
    x = np.arange(1, len(values) + 1) / len(values)
    y = np.cumsum(values) / total_time

    nb_half = np.searchsorted(y, 0.5)
    x_half = (nb_half + 1) / len(values)

    plt.plot(x, y)
    plt.title(f"nodes cumsum (after sorting by decreasing time)\ntotal_time={total_time:.2f}ms")
    plt.ylabel(f"cumsum (in % of total time)")
    plt.xlabel(f"% of nodes")
    plt.axvline(x=x_half, label=f"cumsum=0.5 after {nb_half} nodes", color="r")
    plt.axhline(y=0.5, color="r")
    plt.legend()
    plt.show()


def print_costly_nodes(model, cumsum_thres=0.6):
    timings = model.context.profiler.timing
    timings = {key: np.sum(val) for key, val in timings.items()}
    sorted_key_val = sorted(timings.items(), key=lambda x: x[1], reverse=True)
    values = [x[1] for x in sorted_key_val]
    total_time = np.sum(values)
    relative_cumsum = np.cumsum(values) / total_time
    thres_idx = np.searchsorted(relative_cumsum, cumsum_thres)
    print("\n---- timings ----")
    for key, val in sorted_key_val[:thres_idx]:
        # if not key.startswith("node_of_RAFT"):
        print("  ", key, ":", val, "ms")

    print()


# MAPPING = {
#     "node_of_2520": "node_of_RAFT/BasicEncoder[cnet]/Sequential[layer1]/ResidualBlock[0]/ReLU[relu]_1",
#     "node_of_2526": "",
#     "": "",
#     "": "",
#     "": "",
#     "": "",
#     "": "",
# }

BLOCK_NAMES = [
    "[cnet]",
    "[fnet]",
    "[update_block]",
    "[corr_fn]",
    "[upsampler]",
    "CorrBlockInitializer",
    "bilinear_sample",
]
SUBBLOCK_NAMES = ["BasicEncoder[cnet]", "SepConvGRU[gru]"]


def print_op_names(model):
    timings = model.context.profiler.timing
    for key in sorted(timings):
        print(key)


def print_not_in_names(model):
    timings = model.context.profiler.timing
    timings = {key: np.sum(val) for key, val in timings.items()}
    sorted_key_val = sorted(timings.items(), key=lambda x: x[1], reverse=True)
    for key, val in sorted_key_val:
        if not any(name in key for name in BLOCK_NAMES + SUBBLOCK_NAMES):
            print("  ", key, ":", val, "ms")


def aggregated_time(model):
    timings = model.context.profiler.timing
    timings = {key: np.sum(val) for key, val in timings.items()}
    sorted_key_val = sorted(timings.items(), key=lambda x: x[1], reverse=True)
    values = [x[1] for x in sorted_key_val]
    total_time = np.sum(values)

    block_times = defaultdict(lambda: 0)
    subblock_times = defaultdict(lambda: 0)

    for key, val in sorted_key_val:
        for name in BLOCK_NAMES:
            if name in key:
                block_times[name] += val
        for name in SUBBLOCK_NAMES:
            if name in key:
                subblock_times[name] += val

    print("\nTime spent in each block:")
    for key, val in sorted(block_times.items(), reverse=True, key=lambda x: x[1]):
        print(key, f": {100*val/total_time:.1f} %")

    print(f"==> accounts for {100*sum(block_times.values())/total_time:.2f} % of total time")

    print("\nTime spent in specific subblocks:")
    for key, val in sorted(subblock_times.items(), reverse=True, key=lambda x: x[1]):
        print(key, f": {100*val/total_time:.1f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="backend")
    torch_parser = subparsers.add_parser("torch")
    trt_parser = subparsers.add_parser("trt")
    trt_parser.add_argument("--precision", choices=["fp16", "fp32", "mix"])
    trt_parser.add_argument("--iters", type=int, default=12)
    trt_parser.add_argument("--engine_path")
    trt_parser.add_argument("--name", default="raft-things")
    kwargs = vars(parser.parse_args())
    backend = kwargs.pop("backend")
    if backend == "torch":
        raise NotImplementedError
    else:
        profile_trt(**kwargs)
