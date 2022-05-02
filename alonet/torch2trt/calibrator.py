import os
import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

from aloscene.frame import Frame


class DataBatchStreamer:
    """Streams dataset in batches to the Calibrator

    Parameters
    ----------
        dataset: Iterable dataset.
            Calibration dataset. Should return a tuple of samples. Default None.
        batch_size: (int)
            Streaming batch size. Default 8.
        limit_batches: (int)
            Maximum baches to use.
    
    Attributes
    ----------
        batch_idx: (int)
            Current batch.
        calib_ds: (np.ndarray)
            Batch of data of calibration (size: (batch_size, C, H, W)).
        max_batch: (int)
            Number of batches used for calibration.

    """
    def __init__(
            self,
            dataset=None,
            batch_size=8,
            limit_batches=None,
            ):
        assert isinstance(dataset[0], tuple), "Calibration dataset should return a tuple of images"
        
        self.batch_idx = 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_inputs = len(dataset[0])

        ## instantiate calibration data holders
        shapes = [dataset[0][i].shape[-3:] for i in range(self.n_inputs)]
        self.calib_ds = [np.ones((batch_size, *shapes[i])) for i in range(self.n_inputs)]

        dlength = len(dataset)
        self.max_batch = dlength // batch_size + (1 if dlength % batch_size else 0)
        if limit_batches is not None:
            self.max_batch = min(self.max_batch, limit_batches)
    
    def reset(self):
        """Resets batch index"""
        self.batch_idx = 0
    
    def next_(self):
        """Returns next batch"""
        if self.batch_idx < self.max_batch:
            bidx = self.batch_idx * self.batch_size
            eidx = self.batch_idx * (self.batch_size + 1)
            for i, j in enumerate(range(bidx, eidx)):
                frames = self.dataset[j]
                for k in range(self.n_inputs):
                    frame = frames[k]
                    if isinstance(frame, Frame):
                        frame = frame.as_numpy()
                    self.calib_ds[k][i] = frame

            self.batch_idx += 1
            return [np.ascontiguousarray(self.calib_ds[i], dtype=np.float32) for i in range(self.n_inputs)]
        else:
            return None
    
    def __len__(self):
        return max_batch
    

class BaseCalibrator:
    """Tensorrt post training quantization data calibrator
    
    Parameters
    ----------
        data_streamer: (DataBatchStreamer)
            Data streamer.
        cache_file: (str)
            Path to calibration cache file. Default None.
    
    Attributes
    ----------
        nbytes: List[int]
            List of number of bytes occupied by each dataset batch.

    """
    def __init__(
            self,
            data_streamer,
            cache_file=None,
        ):
        ## Avoid confusing: Deleting calibration file as the read funtion comes first.
        if os.path.exists(cache_file):
            print("Cache file exists already: Deleting file...")
            os.remove(cache_file)
        self.cache_file = cache_file
        self.data_streamer = data_streamer
        self.n_inputs = data_streamer.n_inputs
        self.nbytes = [self.data_streamer.calib_ds[i].nbytes for i in range(self.n_inputs)]
        self.d_input = [cuda.mem_alloc(nbyte) for nbyte in self.nbytes]
        data_streamer.reset()

    def get_batch_size(self):
        return self.data_streamer.batch_size

    def get_batch(self, names):
        batch = self.data_streamer.next_()

        ## return None if the batch is empty
        if batch is None:
            return None
        
        for i in range(self.n_inputs):
            cuda.memcpy_htod(self.d_input[i], batch[i])
        return self.d_input
        
    def read_calibration_cache(self):
        ## expilicitly returns None if cache file does not exist.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        if self.cache_file is None:
            return None

        with open(self.cache_file, 'wb') as f:
            f.write(cache)
    
    def free(self):
        for d in self.d_input:
            d.free()


class MinMaxCalibrator(BaseCalibrator, trt.IInt8MinMaxCalibrator):
    def __init__(
            self,
            data_streamer,
            cache_file,
            **kwargs,
        ):
        trt.IInt8MinMaxCalibrator.__init__(self)
        super(MinMaxCalibrator, self).__init__(data_streamer=data_streamer, cache_file=cache_file, **kwargs)


class LegacyCalibrator(BaseCalibrator, trt.IInt8LegacyCalibrator):
    def __init__(
            self,
            data_streamer,
            cache_file,
            **kwargs,
        ):
        trt.IInt8LegacyCalibrator.__init__(self)
        super(LegacyCalibrator, self).__init__(data_streamer=data_streamer, cache_file=cache_file, **kwargs)


class EntropyCalibrator(BaseCalibrator, trt.IInt8EntropyCalibrator):
    def __init__(
            self,
            data_streamer,
            cache_file,
            **kwargs,
        ):
        trt.IInt8EntropyCalibrator.__init__(self)
        super(EntropyCalibrator, self).__init__(data_streamer=data_streamer, cache_file=cache_file, **kwargs)


class EntropyCalibrator2(BaseCalibrator, trt.IInt8EntropyCalibrator2):
    def __init__(
            self,
            data_streamer,
            cache_file,
            **kwargs,
        ):
        trt.IInt8EntropyCalibrator2.__init__(self)
        super(EntropyCalibrator2, self).__init__(data_streamer=data_streamer, cache_file=cache_file, **kwargs)
