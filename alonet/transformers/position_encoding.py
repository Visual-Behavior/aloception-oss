"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

import aloscene


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, center=False):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.center = center

    def forward(self, ftmap_mask: tuple):
        """

        TODO. We must made one global position encoding in the transformer class
        and one specific class in Detr that inhert from the main psoition embedding method.

        Parameters
        ----------
        ftmap_mask: tuple
            Tuple made of one torch.Tensor (The feature map) and one mask for the transformer
        """
        ft_maps, mask = ftmap_mask
        # mask = torch.squeeze(mask, dim=1)
        # Pytorch will complicate the Squeeze op when exporting to ONNX
        # So  we should use slicing instead of Squeeze
        mask = mask.to(torch.float32)  # TensorRT doesn't support slicing/gathering on bool
        mask = mask[:, 0]
        mask = mask.to(torch.bool)
        # assert mask is not None and mask.dtype == torch.bool

        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            if self.center:
                y_embed = y_embed - 0.5
                x_embed = x_embed - 0.5
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=ft_maps.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


def build_position_encoding():

    position_embedding = "sine"
    hidden_dim = 256

    N_steps = hidden_dim // 2
    if position_embedding in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding
