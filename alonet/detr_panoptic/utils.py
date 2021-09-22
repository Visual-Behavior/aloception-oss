from typing import Dict, List
import torch

import aloscene


def get_mask_queries(
    frames: aloscene.frame, m_outputs: Dict, model: torch.nn, matcher: torch.nn = None, filters: List = None, **kwargs
):

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
        torch.cat([dec_outputs[b : b + 1, idx], torch.zeros(1, max_size - fs, *feat_size, device=device)], dim=1,)
        for b, (idx, fs) in enumerate(zip(filters, fsizes))
    ]
    return torch.cat(dec_outputs, dim=0), filters, kwargs
