import alonet


class CocoPanoptic2Detr(alonet.detr.CocoDetection2Detr):
    def __init__(self, train_stuff_ann: str = None, val_stuff_ann: str = None, **kwargs):
        """CocoDetection2Detr that is used in segmentation applications

        Parameters
        ----------
        train_stuff_ann : str, optional
            Path with stuff annotations for training set, by default None
        val_stuff_ann : str, optional
            Path with stuff annotations for valid set, by default None
        kwargs : dict, optional
            CocoDetection2Detr parameters
        """
        super().__init__(**kwargs)
        self.train_loader_kwargs.update(dict(stuff_ann_file=train_stuff_ann, return_masks=True))
        self.val_loader_kwargs.update(dict(stuff_ann_file=val_stuff_ann, return_masks=True))
        self.val_check()  # Check val loader and set some previous parameters
