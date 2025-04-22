from torchvision.transforms import ColorJitter
import numpy as np

from alodataset.transforms import AloTransform
from aloscene.frame import Frame
import torch


class TemplateTransform(AloTransform):
    def __init__(self, hyper_param, *args, **kwargs):
        self.hyper_param = hyper_param
        super().__init__(*args, **kwargs)

    def sample_params(self):
        self.attribute = "whatever random assignment"
        return (self.attribute,)

    def set_params(self, attribute):
        """No parameters to set"""
        (self.attribute,)

    def apply(self, frame: Frame):
        return frame


class ColorTransform(AloTransform):
    def __init__(self, asymetric_prob=0.2, *args, **kwargs):
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.asymmetric_color_aug_prob = asymetric_prob
        super().__init__(*args, **kwargs)
        assert self.same_on_sequence, "this augmentation is always applied on a sequence of size 2"

    def sample_params(self):
        self.asymetric = np.random.rand() < self.asymmetric_color_aug_prob
        return (self.asymetric,)

    def set_params(self, asymetric):
        """No parameters to set"""
        self.asymetric = asymetric

    def apply(self, frames: Frame):
        assert (
            frames.names[0] == "T" and frames.shape[0] == 2
        ), "this augmentation is always applied on a sequence of size 2"

        n_frames = frames.norm01()

        labels = n_frames.drop_labels()
        names = n_frames.names
        n_frames = n_frames.as_tensor()

        if self.asymetric:
            frame1 = self.photo_aug(n_frames[0])
            frame2 = self.photo_aug(n_frames[1])
            n_frames = torch.stack([frame1, frame2], dim=0)
        else:
            n_frames = self.photo_aug(n_frames)
        n_frames = Frame(n_frames, names=names, normalization="01")
        n_frames.set_labels(labels)

        if n_frames.normalization != frames.normalization:
            n_frames = n_frames.norm_as(frames)
        return frames


class EraserTransform(AloTransform):
    def __init__(self, eraser_aug_prob=0.5, bounds=[50, 100], *args, **kwargs):
        self.eraser_aug_prob = eraser_aug_prob
        self.bounds = bounds
        super().__init__(*args, **kwargs)

    def sample_params(self):
        erase = np.random.rand() < self.eraser_aug_prob
        self.crops = []
        if erase:
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.rand()
                y0 = np.random.rand()
                dx = np.random.randint(self.bounds[0], self.bounds[1])
                dy = np.random.randint(self.bounds[0], self.bounds[1])
                self.crops.append((x0, y0, dx, dy))
        return (self.crops,)

    def set_params(self, crops):
        """No parameters to set"""
        self.crops = crops

    def apply(self, frames: Frame):

        assert tuple(frames.names) in [("C", "H", "W"), ("T", "C", "H", "W")]
        ht, wd = frames.HW
        mean_color = frames.as_tensor().mean(dim=(-2, -1), keepdim=True)
        for x0, y0, dx, dy in self.crops:
            x0 = int(x0 * wd)
            y0 = int(y0 * ht)
            frames[..., y0 : y0 + dy, x0 : x0 + dx] = mean_color
        return frames


class SpatialTransform(AloTransform):
    def __init__(
        self,
        crop_size,
        min_scale=-0.2,
        max_scale=0.5,
        max_stretch=0.2,
        stretch_prob=0.8,
        spatial_aug_prob=0.8,
        h_flip_prob=0.5,
        v_flip_prob=0.1,
        do_flip=True,
        *args,
        **kwargs
    ):
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_stretch = max_stretch
        self.stretch_prob = stretch_prob
        self.spatial_aug_prob = spatial_aug_prob
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.do_flip = do_flip
        super().__init__(*args, **kwargs)
        assert self.same_on_sequence

    def sample_params(self):

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        self.scale_x = scale
        self.scale_y = scale
        if np.random.rand() < self.stretch_prob:
            self.scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            self.scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        self.spatial_aug = np.random.rand() < self.spatial_aug_prob
        self.h_flip = np.random.rand() < self.h_flip_prob if self.do_flip else False
        self.v_flip = np.random.rand() < self.v_flip_prob if self.do_flip else False
        return (self.scale_x, self.scale_y, self.spatial_aug, self.h_flip, self.v_flip)

    def set_params(self, scale_x, scale_y, spatial_aug, h_flip, v_flip):
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.spatial_aug = spatial_aug
        self.h_flip = h_flip
        self.v_flip = v_flip

    def apply(self, frames: Frame):
        ht, wd = frames.HW

        if self.spatial_aug:
            min_scale = np.maximum((self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd))
            scale_x = np.clip(self.scale_x, min_scale, None)
            scale_y = np.clip(self.scale_y, min_scale, None)
            new_w = round(wd * scale_x)
            new_h = round(ht * scale_y)
            frames = frames.resize((new_h, new_w))

        if self.h_flip:
            frames = frames.hflip()

        if self.v_flip:
            frames = frames.vflip()

        return frames
