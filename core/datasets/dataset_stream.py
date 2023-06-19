from typing import Sequence
import numpy as np
import torch


class DatasetStream(torch.utils.data.IterableDataset):
    def __init__(self, dataset, seed=0, shuffle=True):
        assert isinstance(dataset, (Sequence, torch.utils.data.Dataset))
        self.data = dataset
        self.keys = np.arange(len(self.data))
        self.seed = seed
        self.shuffle = shuffle

    def __iter__(self):
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0
        mod = world_size
        shift = rank
        epoch = 0
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            mod *= worker_info.num_workers
            shift = shift * worker_info.num_workers + worker_info.id
        while True:
            if self.shuffle:
                rng = np.random.default_rng(seed=self.seed + epoch)
                rng.shuffle(self.keys)
            for key in self.keys:
                if (key + shift) % mod == 0:
                    yield self.data[key]
            epoch += 1
