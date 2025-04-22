import os
import subprocess
import torch
from torch import nn
import copy
from alonet import ALONET_ROOT


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    y = torch.log(x1 / x2)
    return y
