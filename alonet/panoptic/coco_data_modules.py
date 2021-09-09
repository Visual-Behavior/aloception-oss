import alonet


class CocoPanoptic2Detr(alonet.detr.CocoDetection2Detr):
    def __init__(self, train_stuff_ann: str = None, val_stuff_ann: str = None, **kwargs):
        super().__init__(**kwargs)
        self.train_loader_kwargs.update(dict(stuff_ann_file=train_stuff_ann, return_masks=True))
        self.val_loader_kwargs.update(dict(stuff_ann_file=val_stuff_ann, return_masks=True))
