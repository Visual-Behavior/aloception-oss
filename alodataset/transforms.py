""" Transformation and data augmentation for Frames class from the aloception.scene package
"""
from typing import *
import random
import numpy as np

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.distributions.uniform import Uniform
import torchvision

from aloscene import Frame


class AloTransform(object):
    def __init__(self, same_on_sequence: bool = True, same_on_frames: bool = False):
        """Alo Transform. Each transform in the project should
        inhert from this class.

        Properties
        ----------
        same_on_sequence: bool or float
            Apply the same transformation on each element of the sequences
            If float, between 0 and 1, probability to apply same transformation on each element
        same_on_frames: bool or float
            Apply the same transformations on each frame.
            If float, between 0 and 1, probability to apply same transformation on each frame
        """
        self.same_on_sequence = same_on_sequence
        self.same_on_frames = same_on_frames
        self.sample_params()

    def _init_same_on(self):
        def _prob_to_bool(param):
            if isinstance(param, bool):
                return param
            elif isinstance(param, float):
                if (param < 0) or (param > 1):
                    raise ValueError("Probability value should be between 0 and 1.")
                else:
                    return np.random.rand() < param
            else:
                raise TypeError("param should be bool or float")

        return _prob_to_bool(self.same_on_sequence), _prob_to_bool(self.same_on_frames)

    def sample_params(self):
        raise Exception("Must be implement by a child class")

    def set_params(self):
        raise Exception("Must be implement by a child class")

    def __call__(self, frames: Union[Mapping[str, Frame], List[Frame], Frame], **kwargs):
        """Iter on the given frame(s) or return the frame.
        Based on `same_on_sequence` and  `same_on_frames` parameters
        the method will return and call the `sample_params` method at different time.

        Parameters
        ----------
        frames (dict|list|Frame)
            Could be a dict mapping frame's name to `Frame`, or a list of `Frame`, or a `Frame`.
        """
        seqid2params = {}
        frame_params = None

        same_on_sequence, same_on_frames = self._init_same_on()

        # Go through each image
        if isinstance(frames, dict):

            n_set = {}

            if same_on_sequence is None or same_on_frames is None:
                # TODO: Handle frame sequence only
                raise Exception(
                    "Both `same_on_sequence` and `same_on_frames` must be set if the transformation is called with a dict of frame"
                )

            for key in frames:

                # Go throguh each element of the sequence
                # (If needed to apply save the params for each time step
                if "T" in frames[key].names and same_on_frames and not same_on_sequence:

                    n_set[key] = []
                    for t in range(0, frames[key].shape[0]):
                        if t not in seqid2params:
                            seqid2params[t] = self.sample_params()
                            self.set_params(*seqid2params[t])
                        else:  # Used the params sampled from the previous sequence
                            self.set_params(*seqid2params[t])

                        result = self.apply(frames[key][t], **kwargs)
                        if result.HW != frames[key][t].HW:
                            raise Exception(
                                "Impossible to apply non-deterministic augmentations on the sequence if the augmentations sample different frame size"
                            )

                        n_set[key].append(result.temporal())
                    n_set[key] = torch.cat(n_set[key], dim=0)
                # Different for each element of the sequence, but we don't need to save
                # the params for each image neither
                elif "T" in frames[key].names and not same_on_frames and not same_on_sequence:

                    n_set[key] = []

                    for t in range(0, frames[key].shape[0]):
                        self.set_params(*self.sample_params())
                        result = self.apply(frames[key][t], **kwargs)
                        if result.HW != frames[key][t].HW:
                            raise Exception(
                                "Impossible to apply non-deterministic augmentations on the sequence if the augmentations sample different frame size"
                            )
                        n_set[key].append(result.temporal())
                    n_set[key] = torch.cat(n_set[key], dim=0)
                # Same on all frames
                elif same_on_frames:
                    frame_params = self.sample_params() if frame_params is None else frame_params
                    # print('same_on_frames.....', frame_params)
                    self.set_params(*frame_params)
                    n_set[key] = self.apply(frames[key], **kwargs)

                else:
                    # print("not same on frames")
                    self.set_params(*self.sample_params())
                    n_set[key] = self.apply(frames[key], **kwargs)

            return n_set
        else:
            if "T" in frames.names and not same_on_sequence:
                n_frames = []
                last_size = None
                for t in range(0, frames.shape[0]):
                    frame_params = self.sample_params() if frame_params is None else frame_params
                    self.set_params(*self.sample_params())
                    result = self.apply(frames[t], **kwargs)
                    n_frames.append(result.temporal())
                frames = torch.cat(n_frames, dim=0)
            else:
                self.set_params(*self.sample_params())
                frames = self.apply(frames, **kwargs)

            return frames


class Compose(AloTransform):
    def __init__(self, transforms: AloTransform, *args, **kwargs):
        """Compose a set of transformation

        Parameters
        ----------
        transforms: (list of AloTransform)
            List of transformation to apply sequentially
        """
        self.transforms = transforms
        super().__init__(*args, **kwargs)

    def sample_params(self):
        """Sample and set params of all the child transformations
        into the `self.transforms` list.
        """
        params = []
        for t in self.transforms:
            params.append(t.sample_params())
        return (params,)

    def set_params(self, params):
        """Given predefined params, set the params to all the  child
        transformations.
        """
        for p, t in enumerate(self.transforms):
            t.set_params(*params[p])
        return params

    def apply(self, frame: Frame):
        """Apply the transformation

        Parameters
        ----------
        frame: Frame
            Frame to apply the transformation on
        """
        for t in self.transforms:
            frame = t(frame)
        return frame

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomSelect(AloTransform):
    def __init__(self, transforms1: AloTransform, transforms2: AloTransform, p: float = 0.5, *args, **kwargs):
        """Randomly selects between transforms1 and transforms2,
        with probability p for transforms1 and (1 - p) for transforms2

        Parameters
        ----------
        transforms1: (AloTransform)
            First transformation to apply
        transforms2: (AloTransform)
            Second transformation to apply
        """
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p
        super().__init__(*args, **kwargs)

    def sample_params(self):
        """Sample a `number` between and 1. The first transformation
        will be aply if  `number` is < `self.p` otherwise the second
        transformation is apply.
        """
        self._r = random.random()
        return (self._r, self.transforms1.sample_params(), self.transforms2.sample_params())

    def set_params(self, _r, param1, param2):
        """Given predefined params, set the params on the class"""
        self._r = _r
        self.transforms1.set_params(*param1)
        self.transforms2.set_params(*param2)

    def apply(self, frame: Frame):
        """Apply the transformation

        Parameters
        ----------
        frame: Frame
            Frame to apply the transformation on
        """
        if self._r < self.p:
            return self.transforms1(frame)
        return self.transforms2(frame)


class RandomHorizontalFlip(AloTransform):
    def __init__(self, p: float = 0.5, *args, **kwargs):
        """Randomly apply an horizontal flip on the frame  with
        probability `p`.

        Parameters
        ----------
        p: float
            Probability to apply the transformation
        """
        self.p = p
        super().__init__(*args, **kwargs)

    def sample_params(self):
        """Sample a `number` between and 1. The transformation
        will be aply if  `number` is < `self.p`
        """
        self._r = random.random()
        return (self._r,)

    def set_params(self, _r):
        """Given predefined params, set the params on the class"""
        self._r = _r

    def apply(self, frame: Frame):
        """Apply the transformation

        Parameters
        ----------
        frame: Frame
            Frame to apply the transformation on
        """
        if self._r < self.p:
            return frame.hflip()
        return frame


class RandomSizeCrop(AloTransform):
    def __init__(self, min_size: Union[int, float], max_size: Union[int, float], *args, **kwargs):
        """Randomly crop the frame. The region will be sample
        so that the width & height of the crop will be between
        `min_size` & `max_size`.

        Parameters
        ----------
        min_size: int | float
            Minimum width and height of the crop. I float, will be use as a percentage
        max_size: int  | float
            Maximun width and height of the crop. I float, will be use as a percentage
        """
        if type(min_size) != type(max_size):
            raise Exception("Both `min_size` and `max_size` but be of the same type (float or int)")
        self.min_size = min_size
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def sample_params(self):
        """Sample a `number` between and 1. The transformation
        will be aply if  `number` is < `self.p`
        """
        if isinstance(self.min_size, int):
            self._w = random.randint(self.min_size, self.max_size)
            self._h = random.randint(self.min_size, self.max_size)
        else:
            self._w = np.random.uniform(self.min_size, self.max_size)
            self._h = np.random.uniform(self.min_size, self.max_size)
        return (self._w, self._h)

    def set_params(self, _w, _h):
        """Given predefined params, set the params on the class"""
        self._w = _w
        self._h = _h

    def apply(self, frame: Frame):
        """Apply the transformation

        Parameters
        ----------
        frame: Frame
            Frame to apply the transformation on
        """
        if isinstance(self._w, float):
            sample_w = int(round(self._w * frame.W))
            sample_h = int(round(self._h * frame.H))
        else:
            sample_w = self._w
            sample_h = self._h

        w = min(frame.W, sample_w)
        h = min(frame.H, sample_h)

        region = T.RandomCrop.get_params(frame, [h, w])
        frame = frame.crop(H_crop=(region[0]/frame.H, (region[0]+region[2])/frame.H), W_crop=(region[1]/frame.W, (region[1]+region[3])/frame.W))
        return frame


class RandomSizePad(AloTransform):
    def __init__(self, max_size, frame_size, **kwargs):
        if isinstance(max_size, int):
            max_size = (max_size, max_size)
        self.frame_size = frame_size
        self.max_size = max_size
        self.set_params(*self.sample_params())
        super().__init__(**kwargs)

    def sample_params(self):
        """ """
        h, w = self.frame_size
        # print("hw", h, w, self.max_size)
        pad_width = random.randint(0, max(self.max_size[1] - w, 0))
        pad_height = random.randint(0, max(self.max_size[0] - h, 0))
        # print("pad_width, pad_height", pad_width, pad_height)
        pad_left = random.randint(0, pad_width)
        pad_right = pad_width - pad_left
        print("pad_left", pad_left)
        print("pad_right", pad_right)
        pad_top = random.randint(0, pad_height)
        pad_bottom = pad_height - pad_top
        print("pad_top", pad_top)
        print("pad_bottom", pad_bottom)
        return (pad_left, pad_right, pad_top, pad_bottom)

    def set_params(self, pad_left, pad_right, pad_top, pad_bottom):
        """ """
        self._pad_left = pad_left
        self._pad_right = pad_right
        self._pad_top = pad_top
        self._pad_bottom = pad_bottom

    def __call__(self, frame):

        print((self._pad_top, self._pad_bottom), (self._pad_left, self._pad_right))

        return frame.pad(
            offset_y=(self._pad_top, self._pad_bottom), offset_x=(self._pad_left, self._pad_right), pad_boxes=True
        )


class RandomPad(AloTransform):
    def __init__(self, max_size, frame_size, **kwargs):
        if isinstance(max_size, int):
            max_size = (max_size, max_size)
        self.frame_size = frame_size
        self.max_size = max_size
        self.set_params(*self.sample_params())
        super().__init__(**kwargs)

    def sample_params(self):
        """ """
        h, w = self.frame_size
        pad_width = max(self.max_size[1] - w, 0)
        pad_height = max(self.max_size[0] - h, 0)

        pad_left = random.randint(0, pad_width)
        pad_right = pad_width - pad_left

        pad_top = random.randint(0, pad_height)
        pad_bottom = pad_height - pad_top

        return (pad_left, pad_right, pad_top, pad_bottom)

    def set_params(self, pad_left, pad_right, pad_top, pad_bottom):
        """ """
        self._pad_left = pad_left
        self._pad_right = pad_right
        self._pad_top = pad_top
        self._pad_bottom = pad_bottom

    def __call__(self, frame):
        return frame.pad(
            offset_y=(self._pad_top, self._pad_bottom), offset_x=(self._pad_left, self._pad_right), pad_boxes=True
        )


class RandomCrop(AloTransform):
    def __init__(self, size, *args, **kwargs):
        """Randomly crop the frame.

        Parameters
        ----------
        size : tuple of int
            size (h, w) in pixels of the cropped region
        """
        self.size = size
        super().__init__(*args, **kwargs)

    def sample_params(self):
        """ """
        self.top = np.random.uniform()
        self.left = np.random.uniform()
        return (self.top, self.left)

    def set_params(self, top, left):
        """ """
        self.top = top
        self.left = left

    def apply(self, frame: Frame):
        H, W = frame.HW
        h, w = self.size
        top = int(self.top * (H - h + 1))  # 0 <= top <= H-h
        left = int(self.left * (W - w + 1))  # 0 <= left <= W - w
        frame = F.crop(frame, top, left, h, w)
        return frame


class RandomResizeWithAspectRatio(AloTransform):
    def __init__(self, sizes: list, max_size: int = None, *args, **kwargs):
        """Reszie the given given frame to a sampled `size` from the list of
        given `sizes` so that the largest side is equal to `size` and always < to
        `max_size` (if given).

        Parameters
        ----------
        sizes: list
            List of int. Possible size to sample from
        """
        assert isinstance(sizes, list) or max_size == None
        self.sizes = sizes
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_size_with_aspect_ratio(frame: Frame, size: int, max_size: int = None):
        """Given a `frame` and a `size` this method compute a new size  so that the largest
        side is equal to `size` and always < to `max_size` (if given).

        Parameters
        ----------
        frame : Frame
            Frame to resize. Used only to get the width and the height of the target frame to resize.
        size: int
            Desired size
        max_size: int
            Maximum size of the largest side.
        """
        h, w = frame.H, frame.W

        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def sample_params(self):
        """Sample a `size` from the list of possible `sizes`"""
        # Sample one frame size
        self._size = random.choice(self.sizes)
        return (self._size,)

    def set_params(self, _size):
        """Given predefined params, set the params on the class"""
        self._size = _size

    def apply(self, frame: Frame):
        """Apply the transformation

        Parameters
        ----------
        frame: Frame
            Frame to apply the transformation on
        """
        # Sample one frame size
        size = self.get_size_with_aspect_ratio(frame, self._size, self.max_size)
        # Resize frame
        frame = frame.resize(size)
        return frame


class Resize(AloTransform):
    def __init__(self, size: tuple, *args, **kwargs):
        """Reszie the given frame to the target frame size.

        Parameters
        ----------
        sizes: tuple
            (Height and Width)
        """
        assert isinstance(size, tuple)
        self.size = size
        super().__init__(*args, **kwargs)

    def sample_params(self):
        """Sample a `size` from the list of possible `sizes`"""
        return (self.size,)

    def set_params(self, size):
        """Given predefined params, set the params on the class"""
        self.size = size

    def apply(self, frame: Frame):
        """Apply the transformation

        Parameters
        ----------
        frame: Frame
            Frame to apply the transformation on
        """
        frame = frame.resize(self.size)
        return frame


class Rotate(AloTransform):
    def __init__(self, angle: float, *args, **kwargs):
        """Rotate the given frame using the given rotation angle.

        Parameters
        ----------
        angle: float, between 0 and 360
        """
        assert isinstance(angle, float)
        self.angle = angle
        super().__init__(*args, **kwargs)

    def sample_params(self):
        """Sample an `angle` from the list of possible `angles`"""
        return (self.angle,)

    def set_params(self, angle):
        """Given predefined params, set the params on the class"""
        self.angle = angle

    def apply(self, frame: Frame):
        """Apply the transformation

        Parameters
        ----------
        frame: Frame
            Frame to apply the transformation on
        """
        frame = frame.rotate(self.angle)
        return frame


class RealisticNoise(AloTransform):
    def __init__(self, gaussian_std: float = 0.02, shot_std: float = 0.05, same_on_sequence=False, *args, **kwargs):
        """Add an approximation of a realistic noise to the image.

        More precisely, we add a gaussian noise and a shot noise to the image.

        Parameters
        ----------
        gaussian_std: float
            std of the gaussian noise
        shot_std : float
            std of shot noise before multiplying by squared image per-channel intensity

        Notes
        -----
        More details on gaussian and shot noise : https://en.wikipedia.org/wiki/Image_noise.
        Here, the shot noise is approximated with a gaussian distribution.
        """
        self.gaussian_std = gaussian_std
        self.shot_std = shot_std
        super().__init__(*args, same_on_sequence=same_on_sequence, **kwargs)

    def sample_params(self):
        """No parameters to sample"""
        return tuple()

    def set_params(self):
        """No parameters to set"""
        pass

    def apply(self, frame: Frame):
        n_frame = frame.norm01()

        gaussian_noise = torch.normal(mean=0, std=self.gaussian_std, size=frame.shape, device=frame.device)
        shot_noise = torch.normal(mean=0, std=self.shot_std, size=frame.shape, device=frame.device)
        noisy_frame = n_frame + n_frame * n_frame * shot_noise + gaussian_noise
        noisy_frame = torch.clip(noisy_frame, 0, 1)

        if noisy_frame.normalization != frame.normalization:
            noisy_frame = noisy_frame.norm_as(frame)

        return noisy_frame


class CustomRandomColoring(AloTransform):
    def __init__(self, gamma_r=(0.8, 1.2), brightness_r=(0.5, 2.0), colors_r=(0.5, 1.5), *args, **kwargs):
        """
        Random modification of image colors

        Parameters
        ----------
        gamma_r : tuple
            range of (min, max) values for gamma parameter
        brightness_r : tuple
            range of (min, max) values for brightness parameter
        colors_r : tuple
            range of (min, max) values for colors parameter
        """
        self.gamma_r = gamma_r
        self.brightness_r = brightness_r
        self.colors_r = colors_r
        super().__init__(*args, **kwargs)

    def sample_params(self):
        gamma_min, gamma_max = self.gamma_r
        brightness_min, brightness_max = self.brightness_r
        colors_min, colors_max = self.colors_r

        self.gamma = Uniform(gamma_min, gamma_max).sample()
        self.brightness = Uniform(brightness_min, brightness_max).sample()
        self.colors = Uniform(colors_min, colors_max).sample(sample_shape=(3,))

    def apply(self, frame: Frame):
        assert frame.normalization == "01", "frame should be normalized between 0 and 1 before color modification"

        frame = frame**self.gamma
        frame = frame * self.brightness
        # change color by applying different coefficients to R, G, and B channels
        C = frame.shape[frame.names.index("C")]
        labels = frame.drop_labels()
        for c in range(C):
            frame[frame.get_slices({"C": c})] *= self.colors[c % 3]
        frame.set_labels(labels)
        frame = torch.clip(frame, 0, 1)

        return frame


class SpatialShift(AloTransform):
    def __init__(self, size: tuple, *args, **kwargs):
        """Reszie the given frame to the target frame size.

        Parameters
        ----------
        sizes: tuple
            minimum and maximum size of the image spatial shift
        """
        assert isinstance(size, tuple)
        self.size = size
        super().__init__(*args, **kwargs)

    def sample_params(self):
        """Sample a `size` from the list of possible `sizes`"""
        return (np.random.uniform(self.size[0], self.size[1], 2),)

    def set_params(self, percentage):
        """Given predefined params, set the params on the class"""
        self.percentage = percentage

    def apply(self, frame: Frame):
        """Apply the transformation

        Parameters
        ----------
        frame: Frame
            Frame to apply the transformation on
        """
        n_frame = frame.spatial_shift(self.percentage[0], self.percentage[1])
        return n_frame


class ColorJitter(AloTransform, torchvision.transforms.ColorJitter):
    def __init__(
        self,
        *args,
        brightness: tuple = 0.2,
        contrast: tuple = 0.2,
        saturation: tuple = 0.2,
        hue: tuple = 0.2,
        **kwargs,
    ):
        """Reszie the given frame to the target frame size.

        The forward pass is mostly copy past from orchvision.transforms.ColorJitter.
        Simply inhert from ColorJitter and get adapted to be an AloTransform

        Parameters
        ----------
        brightness (float or tuple of python:float (min, max))
            How much to jitter brightness. brightness_factor is chosen uniformly from
            [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of python:float (min, max))
            How much to jitter contrast. contrast_factor is chosen uniformly from
            [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of python:float (min, max))
            How much to jitter saturation. saturation_factor is chosen uniformly from
            [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of python:float (min, max))
            How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        """
        torchvision.transforms.ColorJitter.__init__(
            self, brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        AloTransform.__init__(self, *args, **kwargs)

    def sample_params(self):
        """Sample a `size` from the list of possible `sizes`"""
        return torchvision.transforms.ColorJitter.get_params(
            brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue
        )

    def set_params(self, *params):
        """Given predefined params, set the params on the class"""
        self.params = params

    def apply(self, frame: Frame):
        """Apply the transformation on the frame

        Parameters
        ----------
        frame: aloscene.Frame

        Returns
        -------
        n_frame: aloscene.Frame
        """

        n_frame = frame.norm01()

        frame_data = n_frame.data.as_tensor()

        for fn_id in self.params[0]:
            if fn_id == 0:
                frame_data = F.adjust_brightness(frame_data, self.params[1])
            elif fn_id == 1:
                frame_data = F.adjust_contrast(frame_data, self.params[2])
            elif fn_id == 2:
                frame_data = F.adjust_saturation(frame_data, self.params[3])
            elif fn_id == 3:
                frame_data = F.adjust_hue(frame_data, self.params[4])

        n_frame.data = frame_data

        if n_frame.normalization != frame.normalization:
            n_frame = n_frame.norm_as(frame)

        return n_frame


class RandomDownScale(AloTransform):
    """
    Downscale the image with random ratio.

    Parameters
    ----------
    min_size (tuple of python:int)
        minimum values (Hmin, Wmin) for downsampled image shape
    preserve_ratio : python:bool
        if true, the aspect ratio of downsample image will be the same as the original.
    """

    def __init__(self, min_size, preserve_ratio=False, *args, **kwargs):
        self.min_size = min_size
        self.preserve_ratio = preserve_ratio
        super().__init__(*args, **kwargs)

    def sample_params(self):
        self.h_coef = np.random.uniform()
        self.w_coef = self.h_coef if self.preserve_ratio else np.random.uniform()
        return (self.h_coef, self.w_coef)

    def set_params(self, h_coef, w_coef):
        """Given predefined params, set the params on the class"""
        self.h_coef = h_coef
        self.w_coef = w_coef

    def apply(self, frame: Frame):
        """Apply the transformation

        Parameters
        ----------
        frame: Frame
            Frame to apply the transformation on
        """
        H, W = frame.HW
        Hmin, Wmin = self.min_size
        # minimum ratio to respect min size (max ratio is 1 to force downsample)
        rh_min = Hmin / H
        rw_min = Wmin / W
        if self.preserve_ratio:
            rh_min = rw_min = max(rw_min, rh_min)
        # actual ratio
        rh = rh_min + self.h_coef * (1 - rh_min)
        rw = rw_min + self.w_coef * (1 - rw_min)
        # new size
        Hnew = int(rh * H)
        Wnew = int(rw * W)
        return frame.resize((Hnew, Wnew))


class RandomDownScaleCrop(Compose):
    """
    Randomly downscale image and crop it.

    For a chosen size of crop, the image is successively:
    - randomly downsampled between original size and crop size
    - cropped to crop size

    Parameters
    ----------
    size (tuple of python:int)
        size of cropped image
    preserve_ratio : python:bool
        if true, the aspect ratio of downsample image will be the same as the original.
    """

    def __init__(self, size, preserve_ratio=False, *args, **kwargs):
        transforms = [RandomDownScale(size, preserve_ratio, *args, **kwargs), RandomCrop(size, *args, **kwargs)]
        super().__init__(transforms, *args, **kwargs)


class DynamicCropTransform(AloTransform):
    """Crop image to target crop size at chosen position."""

    def __init__(self, crop_size, *args, **kwargs):
        assert all([isinstance(s, int) for s in crop_size])
        self.crop_size = crop_size

        super().__init__(*args, **kwargs)

    def sample_params(self):
        return (self.crop_size,)

    def set_params(self, size):
        self.crop_size = size

    def apply(self, frame: Frame, center: Union[Tuple[int, int], Tuple[float, float]] = (0.5, 0.5)):
        """
        center: Coordinate of cropped image center. This coordinate is tuple of int or tuple of float.
                Default: (0.5, 0.5)
        """
        if isinstance(center[0], float):
            center_x = frame.W * center[0]
            center_y = frame.H * center[1]
        else:
            center_x = center[0]
            center_y = center[1]

        crop_h, crop_w = self.crop_size[0], self.crop_size[1]
        left = int(center_x - crop_w / 2)
        top = int(center_y - crop_h / 2)
        right = left + crop_w - 1
        bot = top + crop_h - 1

        if left < 0 or top < 0 or right > (frame.W - 1) or bot > (frame.H - 1):
            raise ValueError(
                f"Crop coordinates out of image border.\
                Image size: {frame.HW}, Crop coordinate (top, left, bot, right): ({top}, {left}, {bot}, {right})"
            )

        return F.crop(frame, top, left, crop_h, crop_w)
