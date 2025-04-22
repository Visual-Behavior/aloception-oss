import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch


import alonet
from alonet.raft.corr import CorrBlock, AlternateCorrBlock
from alonet.raft.update import BasicUpdateBlock
from alonet.raft.extractor import BasicEncoder
from alonet.common.abstract_classes import abstract_attribute, check_abstract_attribute_instanciation, super_new
from alonet.raft.utils.utils import coords_grid, upflow8
from aloscene import Flow, Frame


class RAFTBase(nn.Module):
    """Base Class for RAFT Model (should be subclassed)

    Parameters
    ----------
    fnet :
        feature extractor block
    cnet :
        context network block
    update_block :
        update block
    alternate_corr : bool
        If true, use alternative correlation computation (see RAFT Paper)
    weights : str
        path to a weight file (".pth") or name of stored weights
    device:
        device on which the weights of the model will be loaded
    """

    # should be overriden in subclasses
    hidden_dim = abstract_attribute()
    context_dim = abstract_attribute()
    corr_levels = abstract_attribute()
    corr_radius = abstract_attribute()
    out_plane = abstract_attribute()

    # checks that all abstract attribute are instanciated in child class
    def __new__(cls, *args, **kwargs):
        check_abstract_attribute_instanciation(cls)
        return super_new(RAFTBase, cls, *args, **kwargs)

    def __init__(
        self,
        fnet,
        cnet,
        update_block,
        weights: str = None,
        corr_block=CorrBlock,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.fnet = fnet
        self.cnet = cnet
        self.update_block = update_block
        self.corr_block = corr_block

        if weights is not None:
            weights_from_original_repo = ["raft-things", "raft-chairs", "raft-small", "raft-kitti", "raft-sintel"]
            if weights in weights_from_original_repo or ".pth" in weights or ".ckpt" in weights:
                alonet.common.load_weights(self, weights, device)
            else:
                raise ValueError(f"Unknown weights: '{weights}'")

    @property
    def hdim(self):
        return self.hidden_dim

    @property
    def cdim(self):
        return self.context_dim

    def build_fnet(self, encoder_cls=BasicEncoder, output_dim=256):
        """
        Build RAFT feature extractor
        """
        return encoder_cls(output_dim=output_dim, norm_fn="instance", dropout=self.dropout)

    def build_cnet(self, encoder_cls=BasicEncoder):
        """
        Build RAFT Context Network
        """
        return encoder_cls(output_dim=self.hdim + self.cdim, norm_fn="batch", dropout=self.dropout)

    def build_update_block(self, update_cls=BasicUpdateBlock):
        """
        Build RAFT Update Block
        """
        return update_cls(self.corr_levels, self.corr_radius, hidden_dim=self.hdim, out_planes=self.out_plane)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        if mask is None:
            return upflow8(flow)
        else:
            N, _, H, W = flow.shape
            mask = mask.view(N, 1, 9, 8, 8, H, W)
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(8 * flow, [3, 3], padding=1)
            up_flow = up_flow.view(N, self.out_plane, 9, 1, 1, H, W)

            up_flow = torch.sum(mask * up_flow, dim=2)
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
            return up_flow.reshape(N, self.out_plane, 8 * H, 8 * W)

    def forward_heads(self, m_outputs, only_last=False):
        if not only_last:
            for out_dict in m_outputs:
                out_dict["up_flow"] = self.upsample_flow(out_dict["flow"], out_dict["up_mask"])

        else:
            m_outputs[-1]["up_flow"] = self.upsample_flow(m_outputs[-1]["flow"], m_outputs[-1]["up_mask"])
        return m_outputs

    def forward(self, frame1: Frame, frame2: Frame, iters=12, flow_init=None, only_last=False):
        """Estimate optical flow between pair of frames

        Parameters
        ----------
        frame1 : aloscene.Frame
            frame at time t
        frame2 : aloscene.Frame
            frame at time t+1
        iters : int
            number of iteration of raft update block
        flow_init :
            initial value of flow
        only_last :
            If true, returns the flow of last update block iteration, before and after upsampling.
            If false returns the output flow for all update iterations, after upsampling.

        Returns
        -------
        flows : list of torch.Tensor
            output flows
        """

        assert frame1.normalization == "minmax_sym"
        assert frame2.normalization == "minmax_sym"
        frame1 = frame1.as_tensor()
        frame2 = frame2.as_tensor()

        # run the feature network
        fmap1, fmap2 = self.fnet([frame1, frame2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = self.corr_block(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        cnet = self.cnet(frame1)
        net, inp = torch.split(cnet, [self.hdim, self.cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(frame1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        m_outputs = list()

        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow
            m_outputs.append(
                {"flow": coords1 - coords0, "hidden_state": net, "up_mask": up_mask, "delta_flow": delta_flow}
            )

        return self.forward_heads(m_outputs, only_last=only_last)

    @torch.no_grad()
    def inference(self, m_outputs, only_last=False):
        def generate_frame(out_dict):
            # flow_low = Flow(out_dict["flow"], names=("B", "C", "H", "W"))
            flow_up = Flow(out_dict["up_flow"], names=("B", "C", "H", "W"))
            return flow_up

        if only_last:
            return generate_frame(m_outputs[-1])
        else:
            return [generate_frame(out_dict) for out_dict in m_outputs]


class RAFT(RAFTBase):
    """
    RAFT Model

    Parameters
    ----------
     Parameters
    ----------
    fnet :
        feature extractor block
    cnet :
        context network block
    update_block :
        update block
    alternate_corr : bool
        If true, use alternative correlation computation (see RAFT Paper)
    weights : str
        path to a weight file (".pth") or name of stored weights
    device:
        device on which the weights of the model will be loaded
    dropout : float
        probability for an element to be zero-ed during dropout
    """

    hidden_dim = 128
    context_dim = 128
    corr_levels = 4
    corr_radius = 4
    out_plane = 2

    def __init__(self, dropout=0, **kwargs):
        self.dropout = dropout

        fnet = self.build_fnet(encoder_cls=BasicEncoder, output_dim=256)
        cnet = self.build_cnet(encoder_cls=BasicEncoder)
        update_block = self.build_update_block(update_cls=BasicUpdateBlock)

        super().__init__(fnet, cnet, update_block, **kwargs)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="RAFT inference on two consecutive frames")
    parser.add_argument("image1", type=str, help="Path to the image of the first frame")
    parser.add_argument("image2", type=str, help="Path to the image of the second frame")
    parser.add_argument("--weights", type=str, default="raft-things", help="Name or path to weights file")
    args = parser.parse_args()

    # load data
    device = torch.device("cuda")
    frame1 = Frame(args.image1).batch().norm_minmax_sym().to(device)
    frame2 = Frame(args.image2).batch().norm_minmax_sym().to(device)
    assert frame1.shape == frame2.shape, "flow must be computed between images of same shape"

    # pad frames to multiple of 8
    padder = alonet.raft.utils.Padder()
    frame1 = padder.pad(frame1)
    frame2 = padder.pad(frame2)

    # load raft with "raft-things" weights
    raft = RAFT(weights=args.weights)
    raft.eval()
    raft.to(device)

    # inference
    with torch.no_grad():
        m_outputs = raft.forward(frame1, frame2)  # keep only last stage flow estimation
        output = raft.inference(m_outputs)

        flow = output[-1]
        flow = padder.unpad(flow)  # unpad to original image resolution
        flow = flow.detach().cpu()
        flow.get_view().render()
