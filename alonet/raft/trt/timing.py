from torch.utils.data import SequentialSampler
import numpy as np
import time
import os


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


def load_trt_model(engine_path):
    load_trt_plugins_raft()
    model = TRTExecutor(engine_path)
    model.print_bindings_info()
    return model


def load_torch_model():
    model = RAFT(weights="raft-things")
    model.eval().to("cuda")
    return model


def timing_trt(engine_path=None):
    if engine_path is None:
        engine_path = os.path.join(ALONET_ROOT, "raft-things_fp32.engine")

    model = load_trt_model(engine_path)
    frame1, frame2 = get_sample_images(for_trt=True)

    model.inputs[0].host = frame1
    model.inputs[1].host = frame2

    # GPU warm up
    [model.execute() for i in range(3)]

    # Time engine inference
    N = 20
    times = []
    print("Executing engine for timing...")
    for i in range(N):
        print(f"  forward pass {i+1}/{N}")
        tic = time.time()
        model.execute()
        times.append(time.time() - tic)

    times = (np.array(times) * 1000).astype(np.int)  # ms
    print("Engine execution time:")
    print(f"  min : {times.min()} ms")
    print(f"  median: {int(np.median(times))} ms")
    print(f"  max: {times.max()} ms")


def timing_torch():
    frame1, frame2 = get_sample_images(for_trt=False)
    frame1 = frame1.to("cuda")
    frame2 = frame2.to("cuda")
    model = load_torch_model()

    # GPU Warm-up
    [model(frame1, frame2, only_last=True) for _ in range(3)]

    # Time torch inference
    N = 20
    times = []
    print("Executing torch model for timing...")
    for i in range(N):
        print(f"  forward pass {i+1}/{N}")
        tic = time.time()
        model(frame1, frame2, only_last=True)
        times.append(time.time() - tic)

    times = (np.array(times) * 1000).astype(np.int)  # ms
    print("Torch execution time:")
    print(f"  min : {times.min()} ms")
    print(f"  median: {int(np.median(times))} ms")
    print(f"  max: {times.max()} ms")
    print("\n", times)


if __name__ == "__main__":
    # timing_trt()
    timing_torch()
