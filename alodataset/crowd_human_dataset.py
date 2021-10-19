from pathlib import Path

import torch
import torch.utils.data

import numpy as np
import os
import json

from alodataset import BaseDataset
from alodataset.io import fs
from aloscene import BoundingBoxes2D, Frame, Labels

from shutil import copyfile


class CrowdHumanDataset(BaseDataset):
    def __init__(self, img_folder: str = None, ann_file: str = None, boxes_limit=None, **kwargs):
        """ Init the Crowd Human dataset.

        Parameters
        ----------
        img_folder: str
            Path to the image folder relative to the `dataset_dir` (stored into the aloception config file)
        ann_file: str
            Path to the annotation file relative to the `dataset_dir` (stored into the aloception config file)
        boxes_limit: int | None
            Limit the total number of possible boxes. If not None will select the first `boxes_limit`th wider boxes.
        """
        BaseDataset.__init__(self, name="crowdhuman", **kwargs)
        if self.sample:
            return
        else:
            assert img_folder is not None, "When sample = False, img_folder must be given."
            assert ann_file is not None, "When sample = False, ann_file must be given."

        self._img_folder = img_folder
        self._ann_file = ann_file

        self.img_folder = os.path.join(self.dataset_dir, img_folder, "Images")
        self.ann_file = os.path.join(self.dataset_dir, ann_file)
        # If the current ann file do not exists and we're using a prepared
        # dataset, we'll try to set back the directory based on the original
        # folder
        if not os.path.exists(self.ann_file):
            self.dataset_dir = self.dataset_dir.replace("_prepared", "")
            self.img_folder = os.path.join(self.dataset_dir, img_folder, "Images")
            self.ann_file = os.path.join(self.dataset_dir, ann_file)

        # Setup the class names and the background class ID
        self.labels_names = ["person"]

        self.items = self.load_json_lines(self.ann_file)

        self.bbox_types = ["vbox", "fbox", "hbox"]
        self.boxes_limit = boxes_limit

    def load_json_lines(self, fpath):
        assert os.path.exists(fpath)
        with open(fpath, "r") as fid:
            lines = fid.readlines()
        items = [json.loads(line.strip("\n")) for line in lines]
        return items

    def load_gt(self, dict_input):

        if len(dict_input["gtboxes"]) < 1:
            return [], []

        bboxes = {}
        for bt in self.bbox_types:
            bboxes[bt] = []

        classes = []
        for rb in dict_input["gtboxes"]:
            if rb["tag"] in self.labels_names:
                tag = self.labels_names.index(rb["tag"])
            else:
                tag = -1
            if "extra" in rb:
                if "ignore" in rb["extra"]:
                    if rb["extra"]["ignore"] != 0:
                        tag = -1

            if tag != -1:
                for btype in self.bbox_types:
                    bbox = rb[btype]
                    bbox = np.array(bbox)
                    bbox[2:4] = bbox[0:2] + bbox[2:4]
                    bboxes[btype].append(bbox)
                classes.append(tag)

        return bboxes, classes

    def getitem(self, idx):
        if self.sample:
            return BaseDataset.__getitem__(self, idx)
        
        record = self.items[idx]

        image_id = record["ID"]

        image_path = os.path.join(self.img_folder, image_id + ".jpg")

        frame = Frame(image_path)

        all_boxes, objects_class = self.load_gt(record)

        labels_2d = Labels(objects_class, labels_names=self.labels_names, names=("N"), encoding="id")

        for bbox_type in self.bbox_types:
            boxes = BoundingBoxes2D(
                all_boxes[bbox_type],
                boxes_format="xyxy",
                absolute=True if "_prepared" not in self.dataset_dir else False,
                frame_size=frame.HW if "_prepared" not in self.dataset_dir else None,
                names=("N", None),
                labels=labels_2d,
            )

            if self.boxes_limit is not None and len(boxes) > self.boxes_limit:
                boxes = boxes[torch.argsort(boxes.area(), descending=True)[: self.boxes_limit]]

            frame.append_boxes2d(boxes, bbox_type)

        return frame

    def prepare(self):
        """Prepare the dataset. The human crowd dataset has a lot of huge 4k images that drasticly slow down
        the training. To be more effective, this method will go through all images from the dataset and will
        save a new version of the dataset under `{self.dataset_dir_prepared}`. Once the dataset is prepared,
        the path to the dir in /.aloception/alodataset_config.json will be replace by the new prepared one.

        Notes
        -----
        If the dataset is already prepared, this method will simply check that all file
        are prepared and stored into the prepared folder. Otherwise, if the original directory is no longer
        on the disk, the method will simply use the prepared dir as it is and the prepare step will be skiped.
        """
        from alodataset import transforms as T

        if self.sample is not None:  # Nothing to do. Samples are ready
            return

        if self.dataset_dir.endswith("_prepared") and not os.path.exists(self.dataset_dir.replace("_prepared", "")):
            return

        dataset_dir_name = os.path.basename(os.path.normpath(self.dataset_dir))
        if "_prepared" not in dataset_dir_name:
            wip_dir = f".wip_{dataset_dir_name}_prepared"
            prepared_dir = f"{dataset_dir_name}_prepared"
            img_folder = self.img_folder
            ann_file = self.ann_file
        else:
            wip_dir = f".wip_{dataset_dir_name}"
            prepared_dir = dataset_dir_name
            img_folder = os.path.join(self.dataset_dir.replace("_prepared", ""), self._img_folder, "Images")
            ann_file = os.path.join(self.dataset_dir.replace("_prepared", ""), self._ann_file)

        base_datadir = Path(os.path.normpath(self.dataset_dir)).parent

        # Setup a new directory to work with to prepare the dataset
        n_wip_dir = os.path.join(base_datadir, wip_dir)
        n_wip_dir = os.path.normpath(n_wip_dir)
        # Prepared dir (the one used at the end of the process)
        prepared_dir = os.path.join(base_datadir, prepared_dir)
        prepared_dir = os.path.normpath(prepared_dir)

        # Create wip dir
        if not os.path.exists(n_wip_dir):
            os.makedirs(n_wip_dir)

        p = Path(self.dataset_dir)
        p_parts = list(p.parts)
        p_parts[p_parts.index(dataset_dir_name)] = wip_dir

        n_wip_path = Path(*p_parts)

        # WIP image dir
        tgt_image_dir = os.path.join(n_wip_path, self._img_folder)
        tgt_image_dir = os.path.join(tgt_image_dir, "Images")
        # Final image dir
        final_tgt_image_dir = os.path.join(prepared_dir, self._img_folder)
        final_tgt_image_dir = os.path.join(final_tgt_image_dir, "Images")
        # WIP ann file
        tgt_ann_file = os.path.join(n_wip_dir, self._ann_file)
        # Final ann file
        final_tgt_ann_file = os.path.join(prepared_dir, self._ann_file)

        # Create img dir if it not already exists
        if not os.path.exists(tgt_image_dir):
            os.makedirs(tgt_image_dir)

        for root, subdirs, files in os.walk(img_folder):
            nb_images = len(files)
            for f, f_name in enumerate(files):

                # File aready exists in the destination folder.
                if os.path.exists(os.path.join(tgt_image_dir, f_name)) or os.path.exists(
                    os.path.join(final_tgt_image_dir, f_name)
                ):
                    continue

                frame = Frame(os.path.join(root, f_name))
                if max(frame.shape) > 1333:
                    frame = T.RandomResizeWithAspectRatio([800], max_size=1333)(frame)
                    frame.save(os.path.join(tgt_image_dir, f_name))
                else:
                    copyfile(os.path.join(root, f_name), os.path.join(tgt_image_dir, f_name))

                print(f"Preparing dataset: Saving {f_name}... [{f}/{nb_images}]", end="\r")

        if not os.path.exists(tgt_ann_file) and not os.path.exists(final_tgt_ann_file):
            # Write back the file with all boxes in relative position instead of absolute.
            content = self.load_json_lines(ann_file)
            nb_line = len(content)
            for c in range(len(content)):
                line = content[c]
                frame = Frame(os.path.join(img_folder, line["ID"] + ".jpg"))
                for g in range(len(line["gtboxes"])):
                    hbox = line["gtboxes"][g]["hbox"]
                    fbox = line["gtboxes"][g]["fbox"]
                    vbox = line["gtboxes"][g]["vbox"]
                    line["gtboxes"][g]["hbox"] = [
                        hbox[0] / frame.W,
                        hbox[1] / frame.H,
                        hbox[2] / frame.W,
                        hbox[3] / frame.H,
                    ]
                    line["gtboxes"][g]["fbox"] = [
                        fbox[0] / frame.W,
                        fbox[1] / frame.H,
                        fbox[2] / frame.W,
                        fbox[3] / frame.H,
                    ]
                    line["gtboxes"][g]["vbox"] = [
                        vbox[0] / frame.W,
                        vbox[1] / frame.H,
                        vbox[2] / frame.W,
                        vbox[3] / frame.H,
                    ]
                print(f"Write a new metadata file ....{c}/{nb_line}]", end="\r")
            content = "\n".join([json.dumps(line) for line in content])
            with open(tgt_ann_file, "w") as t:
                t.write(content)

        print("Preparing dataset: Moving the whole structure into the final prepared directory (if needed)")
        fs.move_and_replace(n_wip_dir, prepared_dir)

        self.set_dataset_dir(prepared_dir)
        self.ann_file = final_tgt_ann_file
        self.img_folder = final_tgt_image_dir
        self.items = self.load_json_lines(self.ann_file)


def main():
    """ Main
    """
    crowd_human_dataset = CrowdHumanDataset(sample=True)

    crowd_human_dataset.prepare()
    for i, frames in enumerate(crowd_human_dataset.train_loader(batch_size=2, sampler=None, num_workers=0)):
        frames = Frame.batch_list(frames)
        frames.get_view().render(figsize=(20, 10))


if __name__ == "__main__":
    main()
