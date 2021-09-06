from alodataset import WaymoDataset, Split

waymo_dataset = WaymoDataset()
waymo_dataset.prepare()
print("prepared dataset directory:", waymo_dataset.dataset_dir)