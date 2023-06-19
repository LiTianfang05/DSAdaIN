import os
from core.datasets import LMDBDataset, ImageFolder, DatasetStream


def get_dataset(root, transform, seed=None, stream=False):
    if "lsun" in root:
        dataset = LMDBDataset(root, transform)
    elif os.path.isdir(root):
        dataset = ImageFolder(root, transform)
    else:
        raise NotImplementedError
    if stream:
        dataset = DatasetStream(dataset, seed)
    return dataset
