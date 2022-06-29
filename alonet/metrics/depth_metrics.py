import aloscene

import torch
import numpy as np
from typing import List
from alonet.metrics.utils import _print_body, _print_head, _print_map


class DepthMetrics:
    """Computes depth metrics

    Parameters
    ----------
        x : List[int]
            Permissivety levels. Default [1, 2, 3].
        alpha : float
            Permissivety percentage. Default 1.25 (25%).
    
    Raises
    ------
        AssertionError
            x not in [1, 2, 3]
    
    """
    def __init__(
            self,
            x: List[int] = [1, 2, 3],
            alpha: float = 1.25
            ):
        assert isinstance(x, list)
        for xi in x:
            assert isinstance(xi, int)
        
        self.x = sorted(x)
        self.alpha = alpha
        self.metrics = {"RMSE": [], "RMSE_log": [], "Log10": [], "AbsRel": []}

        dx = {f"d{i}": [] for i in x}
        self.metrics.update(dx)
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.metrics.items()}
    
    def __iadd__(self, di):
        if not all(k in self.metrics.keys() for k in di.keys()):
            raise KeyError

        assert all(isinstance(i, type(list(di.values())[0])) or i == np.inf for i in di.values()), "Values instances are not the same"
        if isinstance(list(di.values())[0], list):
            assert all(len(i) == len(list(di.values())[0]) for i in di.values()), "values lengths are not the same"
        
        for k, v in di.items():
            if isinstance(v, float):
                self.metrics[k].append(v)
            elif isinstance(v, list):
                self.metrics[k] +=  v
            else:
                raise Exception(f"Expected values to be list or float, got {v.__class__.__name__} instead")
        
        max_ = max([len(i) for i in self.metrics.values()])
        for k, v in self.metrics.items():
            if len(v) < max_:
                self.metrics[k] += [np.nan] * (max_ - len(v))
        
    def __len__(self):
        return len(self.metrics["RMSE"])
        
    def add_sample(
            self,
            p_depth: aloscene.Depth,
            t_depth: aloscene.Depth,
            mask: aloscene.Mask = None,
            epsilon: float = 1e-5
            ):
        """Computes sample depth metrics

        Parameters
        ----------
            t_depth : aloscene.Depth
                ground truth depth.
            p_depth : aloscene.Depth
                predicted depth.
            mask : Union[aloscene.Mask, np.ndarray]
                mask.
            epsilon : float
                Value to avoid zero division.
        
        """
        metrics = {}
        assert t_depth.shape == p_depth.shape, "Input depths must have the same dimensions"

        p_depth = p_depth.to(torch.device("cpu")).as_numpy()
        t_depth = t_depth.to(torch.device("cpu")).as_numpy()

        p_depth = np.squeeze(p_depth)
        t_depth = np.squeeze(t_depth)

        if mask is None:
            t_depth = self.set_values(t_depth)
            p_depth = self.set_values(p_depth)
        else:
            if isinstance(mask, aloscene.Mask):
                mask = mask.as_numpy().astype(int)

            # FIXME : MASK WITH PR #194            
            assert all(i in [0, 1] for i in np.unique(mask)), "unvalid mask"

            t_depth = t_depth * mask
            p_depth = p_depth * mask

        # dx scores
        ratio = np.maximum(p_depth / (t_depth + epsilon), t_depth / (p_depth + epsilon))

        for xi in self.x:
            th = self.alpha ** xi
            dx = (ratio < th).mean()
            metrics[f"d{xi}"] = float(dx)
        
        # RMSE
        rmse = (p_depth - t_depth) ** 2
        rmse = np.sqrt(rmse.mean())
        metrics["RMSE"] = float(rmse)

        # ABS REL
        abs_rel = (np.abs(t_depth - p_depth) / (t_depth + epsilon)).mean()
        metrics["AbsRel"] = float(abs_rel)

        # RMSE LOG
        rmse_log = (np.log(p_depth + epsilon) - np.log(t_depth + epsilon)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())
        metrics["RMSE_log"] = float(rmse_log)

        # LOG 10
        log10 = np.abs(np.log10(p_depth + epsilon) - np.log10(t_depth + epsilon)).mean()
        metrics["Log10"] = float(log10)

        self += metrics

    @staticmethod
    def set_values(depth: np.ndarray, fillnan: float = 0., fillinf: float = 0.):
        """Sets inf and Nan values to custom value
        
        Parameters
        ----------
            fillnan:
                Value to replace Nan.
            fillinf:
                value to replace Inf.
        
        """
        depth[np.isnan(depth)] = fillnan
        depth[np.isinf(depth)] = fillinf
        return depth
    
    def calc_map(self, print_result: bool = False):
        """Prints depth metrics
        
        """
        for k, v in self.metrics.items():
            v_ = [i for i in v if not np.isnan(i)]
            self.metrics[k] = v_

        scores = {k: np.mean(v) for k, v in self.metrics.items()}
        scores["n"] = len(self)

        hdr = "{:>9} " * len(self.metrics)
        res = "{:>9.3f} " * len(self.metrics)

        clm_size = 11
        _print_head(head_elm=self.metrics.keys(), clm_size=clm_size)
        _print_body(average_pq=scores, pq_per_class=None, clm_size=clm_size)
        return scores
