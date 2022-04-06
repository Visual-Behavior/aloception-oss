from aloscene.tensors.augmented_tensor import AugmentedTensor


class Pose(AugmentedTensor):
    """Pose Tensor.

    Parameters
    ----------
    x: torch.Tensor
        Pose matrix
    """

    @staticmethod
    def __new__(cls, x, *args, names=(None, None), **kwargs):
        tensor = super().__new__(cls, x, *args, names=names, **kwargs)
        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)
