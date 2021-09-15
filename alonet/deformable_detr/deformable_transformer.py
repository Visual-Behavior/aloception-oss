# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from .utils import inverse_sigmoid
from .ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    """Transformer with Multiscale Deformable Attention

    For more details: Deformable DETR https://arxiv.org/abs/2010.04159

    Notes
    -----
    This class can only be used with "cuda" device.
    """

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        encoder=None,
        decoder=None,
        decoder_layer=None,
        encoder_layer=None,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        two_stage=False,
        two_stage_num_proposals=300,
    ):
        """Init Deformable Transformer

        Parameters
        ----------
        d_model : int, optional
            dimension of encoder and decoder, by default 256
        nhead : int, optional
            number of multi-head attention, by default 8
        num_encoder_layers : int, optional
            number of encoder layers, by default 6
        num_decoder_layers : int, optional
            number of decoder layers, by default 6
        dim_feedforward : int, optional
            feedforward dimension in each encoder/decoder block, by default 1024
        dropout : float, optional
            dropout rate, by default 0.1
        activation : str, optional
            activation function, by default "relu"
        return_intermediate_dec : bool, optional
            returns decoder outputs of intermediate layers, by default False
        num_feature_levels : int, optional
            Number of feature maps sampled by multiscale deformable attention, by default 4
        dec_n_points : int, optional
            Number of sampling points in deformable attention in decoder, by default 4
        enc_n_points : int, optional
            Number of sampling points in deformable attention in encoder, by default 4
        two_stage : bool, optional
            Enable two-stage Deformable DETR. See the paper for more detail, by default False
        two_stage_num_proposals : int, optional
            Number of region proposals at first stage, by default 300
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        if encoder is None:
            encoder_layer = encoder_layer or DeformableTransformerEncoderLayer(
                d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
            )
            self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        else:
            self.encoder = encoder

        if decoder is None:
            decoder_layer = decoder_layer or DeformableTransformerDecoderLayer(
                d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points
            )
            self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        else:
            self.decoder = decoder

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        """
        Get the ratio of (active mask shape / mask shape).
        This ratio will be use to create refenrence points for sampling
        to ensure those points in active zone.

        Parameters
        ----------
        mask: (b, H, W)

        Returns
        -------
        valid_ratio: (b, 2)
        """
        H, W = mask.shape[-2], mask.shape[-1]
        valid_H = torch.sum((~mask).float()[:, :, 0], 1, keepdims=True)  # (b, 1)
        valid_W = torch.sum((~mask).float()[:, 0, :], 1, keepdims=True)  # (b, 1)
        valid_ratio_h = valid_H / H
        valid_ratio_w = valid_W / W
        valid_ratio = torch.cat([valid_ratio_w, valid_ratio_h], 1)  # (b, 2)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, **kwargs):
        transformer_outputs = {}

        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = torch.tensor([[h, w]], dtype=torch.long, device=srcs[0].device)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.cat(spatial_shapes, dim=0)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        _valid_ratios = [self.get_valid_ratio(m) for m in masks]
        valid_ratios = torch.stack(_valid_ratios, 1)

        # encoder
        memory = self.encoder(
            src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, **kwargs
        )

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            )

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        # tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
        dec_outputs = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_pos=query_embed,
            src_padding_mask=mask_flatten,
            **kwargs,
        )

        # print("init_reference_out", init_reference_out)
        # print(' dec_outputs["init_reference_out"]', dec_outputs["init_reference_out"])

        dec_outputs
        transformer_outputs.update(dec_outputs)
        # transformer_outputs["init_reference_out"] = init_reference_out  # dec_outputs["init_reference_out"]

        if self.two_stage:
            transformer_outputs["enc_outputs_class"] = enc_outputs_class
            transformer_outputs["enc_outputs_coord_unact"] = enc_outputs_coord_unact
        else:
            transformer_outputs["enc_outputs_class"] = None
            transformer_outputs["enc_outputs_coord_unact"] = None

        return transformer_outputs


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None, **kwargs):
        # self attention
        pos_embed_src = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            pos_embed_src, reference_points, src, spatial_shapes, level_start_index, padding_mask, **kwargs
        )
        _src = src + self.dropout1(src2)
        _src = self.norm1(_src)

        # ffn
        src = self.forward_ffn(_src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get the reference points used in sampling

        Parameters
        ----------
        spatial_shapes: (num_levels, 2)
            height and width of each feature level
        valid_ratios: (b, num_levels, 2)
            ratio: [unpadded height / mask height, unpadded width / mask width]

        Returns
        -------
        reference_points: (b, num_points, num_levels, 2)
            num_points = sum of height*width for each feature levels
        """
        reference_points_list = []
        valid_ratios = torch.unsqueeze(valid_ratios, dim=1)  # (b, 4, 2) -> (b, 1, 4, 2)
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            # ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            #                               torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            # Avoid using linspace because it's not supported in ONNX
            # arange in TensorRT use INT32 only
            range_y = torch.arange(int(H_), dtype=torch.int32, device=device).float() + 0.5
            range_x = torch.arange(int(W_), dtype=torch.int32, device=device).float() + 0.5
            ref_y, ref_x = torch.meshgrid(range_y, range_x)
            ref_y = ref_y.reshape((1, -1))  # (1, H_ * W_)
            ref_x = ref_x.reshape((1, -1))  # (1, H_ * W_)
            # valid_ratios_y = valid_ratios[:, :, lvl, 1]
            # valid_ratios_x = valid_ratios[:, :, lvl, 0]
            valid_ratios_y = valid_ratios[:, :, lvl]
            valid_ratios_x = valid_ratios[:, :, lvl]
            valid_ratios_y = valid_ratios_y[:, :, 1]
            valid_ratios_x = valid_ratios_x[:, :, 0]
            ref_y = ref_y / (valid_ratios_y * H_)  # (1, n) / (b, 1) -> (b, n)
            ref_x = ref_x / (valid_ratios_x * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, **kwargs):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for i, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask, **kwargs)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self, d_model=256, dim_feedforward=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def pre_process_tgt(self, tgt, query_pos, tgt_key_padding_mask, **kwargs):
        """Pre process decoder inputs"""
        return tgt, query_pos, tgt_key_padding_mask

    def decoder_layer_forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        tgt_key_padding_mask=None,
        src_padding_mask=None,
        **kwargs,
    ):
        """Decoder forward layer"""
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), key_padding_mask=tgt_key_padding_mask
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
            **kwargs,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        tgt_key_padding_mask=None,
        src_padding_mask=None,
        **kwargs,
    ):

        tgt, query_pos, tgt_key_padding_mask = self.pre_process_tgt(tgt, query_pos, tgt_key_padding_mask, **kwargs)

        tgt = self.decoder_layer_forward(
            tgt=tgt,
            query_pos=query_pos,
            reference_points=reference_points,
            src=src,
            src_spatial_shapes=src_spatial_shapes,
            level_start_index=level_start_index,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_padding_mask=src_padding_mask,
            **kwargs,
        )

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def pre_process_tgt(self, tgt, query_pos, tgt_key_padding_mask, **kwargs):
        """Pre process decoder inputs"""
        return tgt, query_pos, tgt_key_padding_mask

    def decoder_forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        tgt_key_padding_mask=None,
        **kwargs,
    ):

        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            output = layer(
                tgt=output,
                query_pos=query_pos,
                reference_points=reference_points_input,
                src=src,
                src_spatial_shapes=src_spatial_shapes,
                level_start_index=src_level_start_index,
                src_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                **kwargs,
            )

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        tgt_key_padding_mask=None,
        decoder_outputs: dict = None,
        **kwargs,
    ):

        decoder_outputs = {} if decoder_outputs is None else decoder_outputs

        tgt = tgt.transpose(0, 1)
        query_pos = query_pos.transpose(0, 1)
        tgt, query_pos, tgt_key_padding_mask, reference_points = self.pre_process_tgt(
            tgt, query_pos, tgt_key_padding_mask=tgt_key_padding_mask, reference_points=reference_points, **kwargs
        )
        tgt = tgt.transpose(1, 0)
        query_pos = query_pos.transpose(1, 0)

        output, inter_reference_points = self.decoder_forward(
            tgt=tgt,
            reference_points=reference_points,
            src=src,
            src_spatial_shapes=src_spatial_shapes,
            src_level_start_index=src_level_start_index,
            src_valid_ratios=src_valid_ratios,
            query_pos=query_pos,
            src_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            **kwargs,
        )

        decoder_outputs["init_reference_out"] = reference_points
        decoder_outputs.update({"hs": output, "inter_references_out": inter_reference_points})

        return decoder_outputs


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
    )
