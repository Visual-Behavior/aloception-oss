from typing import Dict, List, Union
import torch

import aloscene


def get_mask_queries(
    frames: aloscene.frame, m_outputs: Dict, model: torch.nn, matcher: torch.nn = None, filters: List = None, **kwargs
):
    """Mask process filter throught matcher or our_filter function

    Parameters
    ----------
    frames : aloscene.frame
        Input frames
    m_outputs : Dict
        Forward output
    model : torch.nn
        model with inference function
    matcher : torch.nn, optional
        Matcher between GT and pred elements, by default None
    filters : List, optional
        Boolean mask for each batch, by default None

    Returns
    -------
    torch.Tensor, List
        Mask reduced from (M,H,W) to (N,H,W) with boolean mask per batch (M >= N)
    """
    dec_outputs = m_outputs["dec_outputs"][-1]
    device = dec_outputs.device
    if filters is None:
        if matcher is None:
            filters = model.get_outs_filter(m_outputs=m_outputs, **kwargs)
        else:
            nq = dec_outputs.size(1)
            filters = [torch.tensor([False] * nq, dtype=torch.bool, device=device) for _ in range(len(dec_outputs))]
            for b, (src, _) in enumerate(matcher(m_outputs=m_outputs, frames=frames, **kwargs)):
                filters[b][src] = True

    # Filter masks and cat adding pad
    fsizes = [sum(f) for f in filters]
    max_size = max(fsizes)
    feat_size = dec_outputs.shape[2:]

    dec_outputs = [
        torch.cat([dec_outputs[b : b + 1, idx], torch.zeros(1, max_size - fs, *feat_size, device=device)], dim=1)
        for b, (idx, fs) in enumerate(zip(filters, fsizes))
    ]
    return torch.cat(dec_outputs, dim=0), filters


def get_base_model_frame(frames: Union[list, aloscene.Frame], cat: str = "category") -> aloscene.Frame:
    """Get frames with correct labels for criterion process

    Parameters
    ----------
    frames : aloscene.Frame
        frames to set labels

    Returns
    -------
    aloscene.Frame
        frames with correct set of labels
    """
    if isinstance(frames, list):
        frames = aloscene.Frame.batch_list(frames)

    frames = frames.clone()

    def criterion(b):
        b.labels = b.labels[cat]

    if isinstance(frames.boxes2d[0].labels, dict):
        frames.apply_on_child(frames.boxes2d, criterion)
    if isinstance(frames.segmentation[0].labels, dict):
        frames.apply_on_child(frames.segmentation, criterion)
    return frames
