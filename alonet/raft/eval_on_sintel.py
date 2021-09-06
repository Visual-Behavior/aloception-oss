from torch.utils.data import SequentialSampler
import numpy as np
import argparse
import torch

from alonet.common.pl_helpers import load_training
from alodataset import SintelDataset, Split
from alonet.raft.utils import Padder
from alonet.raft import LitRAFT
from aloscene import Frame


def sintel_transform_fn(frame):
    return frame["left"].norm_minmax_sym()


def parse_args():
    parser = argparse.ArgumentParser(description="Raft evaluation on Sintel training set")
    parser.add_argument("--weights", default="raft-things", help="name or path to weights file")
    parser.add_argument(
        "--project_run_id", default="raft", help="project run_id (if loading weights from a previous training)"
    )
    parser.add_argument("--run_id", help="load weights from previous training with this run_id")
    args = parser.parse_args()
    if args.run_id is not None:
        args.weights = None
    return args


if __name__ == "__main__":

    args = parse_args()

    model = load_training(LitRAFT, args, no_exception=True).model
    model = model.eval().to("cuda")

    dataset = SintelDataset(
        split=Split.TRAIN,
        cameras=["left"],
        labels=["flow"],
        passes=["clean"],
        sequence_size=2,
        transform_fn=sintel_transform_fn,
    )

    padder = Padder()  # pads inputs to multiple of 8

    with torch.no_grad():
        epe_list = []
        for idx, frames in enumerate(dataset.train_loader(batch_size=1, sampler=SequentialSampler)):
            if idx % 10 == 0:
                print(f"{idx+1:04d}/{len(dataset)}", end="\r")

            frames = Frame.batch_list(frames)
            frame1 = frames[:, 0, ...].clone()
            frame2 = frames[:, 1, ...].clone()
            frame1 = padder.pad(frame1).cuda()
            frame2 = padder.pad(frame2).cuda()

            _, flow_pred = model(frame1, frame2, iters=32, only_last=True)
            flow_pred = flow_pred.cpu()
            flow_pred = padder.unpad(flow_pred)[0]
            flow_gt = frames[0][0].flow["flow_forward"].as_tensor()

            epe = torch.sum((flow_pred - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe = np.mean(np.concatenate(epe_list))
        print("\n\nValidation Sintel EPE: %f" % epe)
