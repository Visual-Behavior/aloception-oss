import torch
import os

from alodataset import BaseDataset, Split, SplitMixin

from aloscene import Frame, Flow, Mask


class ChairsSDHomDataset(BaseDataset, SplitMixin):

    SPLIT_FOLDERS = {Split.VAL: "val", Split.TRAIN: "train"}

    def __init__(self, **kwargs):
        super(ChairsSDHomDataset, self).__init__(name="chairsSDHom", **kwargs)
        if self.sample:
            return
        self.dir_path = self._dir_path()
        self.items = self.get_sequences()

    def _dir_path(self):
        return os.path.join(self.dataset_dir, self.get_split_folder())

    def get_file_ids(self):
        """
        Parse dataset directory to find ids of data samples
        """
        dir_path = self.dir_path
        file_ids = [fname.split("-")[0] for fname in os.listdir(dir_path) if ".flo" in fname]
        return sorted(file_ids)

    def get_sequences(self):
        """
        Returns dict containing all data filepaths for each sequence
        """
        dir_path = self.dir_path
        sequences = {}
        for idx, fid in enumerate(self.get_file_ids()):
            sequences[idx] = {
                "image_0": os.path.join(dir_path, f"{fid}-img_0.png"),
                "image_1": os.path.join(dir_path, f"{fid}-img_1.png"),
                "flow": os.path.join(dir_path, f"{fid}-flow_01.flo"),
                "flow_occ": os.path.join(dir_path, f"{fid}-occ_01.png"),
            }
        return sequences

    def get_frames(self, sequence_data):
        """
        Load frames corresponding to a sequence
        """
        frame_0 = Frame(sequence_data["image_0"]).temporal()
        occ_0 = Mask(sequence_data["flow_occ"])
        flow_0 = Flow(sequence_data["flow"], occlusion=occ_0)
        frame_0.append_flow(flow_0, "flow_forward")
        frame_1 = Frame(sequence_data["image_1"]).temporal()
        frames = torch.cat([frame_0, frame_1], dim=0).type(torch.float32)
        return frames

    def getitem(self, idx):
        """
        Return frames corresponding to a sequence
        """
        if self.sample:
            return BaseDataset.__getitem__(self, idx)
        sequence_data = self.items[idx]
        return self.get_frames(sequence_data)


if __name__ == "__main__":
    dataset = ChairsSDHomDataset(sample=True)
    for idx, frame in enumerate(dataset.stream_loader()):
        frame.get_view().render()
        if idx == 3:
            break
