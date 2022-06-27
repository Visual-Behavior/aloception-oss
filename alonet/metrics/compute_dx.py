import torch
import numpy as np


class DxScoreMetric:
    """Computs depth dx score

    Parameters
    ----------
        x : List[int]
            permissivety levels. Default [1, 2, 3].
        alpha : float
            permissivety percentage. Default 1.25 (25%).
    
    Raises
    ------
        AssertionError
            x not in [1, 2, 3]
    
    """
    def __init__(self, x=[1, 2, 3], alpha=1.25):
        assert isinstance(x, list)
        for xi in x:
            assert isinstance(xi, int)
        
        self.x = sorted(x)
        self.dx = {f"d{i}": [] for i in x}
        self.alpha = alpha
    
    def __getitem__(self, idx):
        return  self.dx[idx]
    
    def __iadd__(self, di):
        if sorted(self.dx.keys()) != sorted(di.keys()):
            raise KeyError("keys are not compatible")
        
        assert all(isinstance(i, type(list(di.values())[0])) for i in di.values()), "Values are not of the same instance"

        for k, v in di.items():
            if isinstance(v, float):
                self.dx[k].append(v)
            elif isinstance(v, list):
                self.dx[k] +=  v
            else:
                raise Exception(f"Expected values to be list or float, got {v.__class__.__name__} instead")
    
    def add_score(self, t_depth, p_depth, mask=None):
        """Computes dx scores

        Parameters
        ----------
            t_depth : aloscene.Depth
                ground truth depth.
            p_depth : aloscene.Depth
                predicted depth.
        
        """
        assert t_depth.shape == p_depth.shape, "Input depths must have the same dimensions"

        p_depth = p_depth.to(torch.device("cpu")).detach().numpy()
        t_depth = t_depth.to(torch.device("cpu")).detach().numpy()
        
        if mask is None:
            t_depth = self.set_values(t_depth)
            p_depth = self.set_values(p_depth)
        else:
            t_depth = t_depth[mask]
            p_depth = p_depth[mask]

        ratio = np.maximum(p_depth / t_depth, t_depth / p_depth)

        for xi in self.x:
            th = self.alpha ** xi
            dx = (ratio < th).mean()
            self.dx[f"d{xi}"].append(dx)

    @staticmethod
    def set_values(depth, fillnan=0., fillinf=0.):
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
    
    def log_scores(self):
        """Prints dx scores
        
        """
        scores = [np.mean(d) * 100 for d in self.dx.values()]

        hdr = "{:>7} " * len(self.dx)
        res = "{:>7.2f} " * len(self.dx)

        print(hdr.format(*self.dx.keys()))
        print(res.format(*scores))
