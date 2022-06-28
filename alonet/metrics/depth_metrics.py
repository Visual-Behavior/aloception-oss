import torch
import numpy as np


class DepthMetrics:
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
        
    def add_score(self, t_depth, p_depth, mask=None):
        """Computes dx scores

        Parameters
        ----------
            t_depth : aloscene.Depth
                ground truth depth.
            p_depth : aloscene.Depth
                predicted depth.
        
        """
        metrics = {}
        assert t_depth.shape == p_depth.shape, "Input depths must have the same dimensions"

        p_depth = p_depth.to(torch.device("cpu")).detach().numpy()
        t_depth = t_depth.to(torch.device("cpu")).detach().numpy()
        
        if mask is None:
            t_depth = self.set_values(t_depth)
            p_depth = self.set_values(p_depth)
        else:
            t_depth = t_depth[mask]
            p_depth = p_depth[mask]

        # dx scores
        ratio = np.maximum(p_depth / t_depth, t_depth / p_depth)

        for xi in self.x:
            th = self.alpha ** xi
            dx = (ratio < th).mean()
            metrics[f"d{xi}"] = float(dx)
        
        # RMSE
        rmse = (p_depth - t_depth) ** 2
        rmse = np.sqrt(rmse.mean())
        metrics["RMSE"] = float(rmse)

        # RMSE LOG
        rmse_log = (np.log(p_depth) - np.log(t_depth)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())
        metrics["RMSE_log"] = float(rmse_log)

        # ABS REL
        abs_rel = (np.abs(t_depth - p_depth) / t_depth).mean()
        metrics["AbsRel"] = float(abs_rel)

        # LOG 10
        log10 = np.abs(np.log10(p_depth) - np.log10(t_depth)).mean()
        metrics["Log10"] = float(log10)

        self += metrics

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
        """Prints depth metrics
        
        """
        for k, v in self.metrics.items():
            v_ = [i for i in v if not np.isnan(i)]
            self.metrics[k] = v_

        scores = [np.mean(d) * 100 for d in self.metrics.values()]

        hdr = "{:>8} " * len(self.metrics)
        res = "{:>8.2f} " * len(self.metrics)

        print(hdr.format(*self.metrics.keys()))
        print(res.format(*scores))
