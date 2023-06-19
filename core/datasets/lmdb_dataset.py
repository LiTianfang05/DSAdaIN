from torchvision.datasets import LSUNClass


class LMDBDataset(LSUNClass):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image
