import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


class AFHQ(datasets.ImageFolder):
    def __init__(self, root, transform, split="train", subset="all"):
        root = os.path.join(root, "afhq")
        assert os.path.exists(root), f"{root} does not exist!"
        assert split in ("train", "val")
        super().__init__(os.path.join(root, split), transform)


class AnimalFaces(datasets.ImageFolder):
    def __init__(self, root, transform=None, subset=None, train=True, num_val=0):
        super().__init__(root, transform)
        self.train = train
        self.min_data = 99999
        self.max_data = -1
        if subset is None:
            self.class_table = tuple(self.class_to_idx.values())
        else:
            self.class_table = {}
            for i, target in enumerate(subset):
                if isinstance(target, str):
                    target = self.class_to_idx[target]
                self.class_table[target] = i
            new_samples, new_targets = [], []
            for k in self.class_table.keys():
                samples = list(filter(lambda s: s[1] == k, self.samples))
                targets = list(filter(lambda y: y == k, self.targets))
                self.min_data = min(self.min_data, len(samples))
                self.max_data = max(self.max_data, len(samples))
                if num_val <= 0:
                    new_samples += samples
                    new_targets += targets
                elif train:
                    new_samples += samples[:-num_val]
                    new_targets += targets[:-num_val]
                else:
                    new_samples += samples[-num_val:]
                    new_targets += targets[-num_val:]
        self.samples = new_samples
        self.targets = new_targets
        for k in self.class_table.keys():
            targets = list(filter(lambda y: y == k, self.targets))
            self.min_data = min(self.min_data, len(targets))
            self.max_data = max(self.max_data, len(targets))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.class_table[target]


class CelebAMaskHQ(Dataset):
    def __init__(self, root, transform, split="train", subset="all"):
        super().__init__()
        root = os.path.join(root, "CelebAMask-HQ")
        assert os.path.exists(root), f"{root} does not exist!"
        self.root = os.path.join(root, "CelebA-HQ-img/")
        self.transform = transform
        ann_path = os.path.join(root, "CelebAMask-HQ-attribute-anno.txt") 
        with open(ann_path, "r") as f:
            lines = f.readlines()[1:]
            lines = [*map(lambda x: x.replace("\n", ""), lines)]
        self.attribute_names = lines[0].split(" ")
        self.samples = []
        for line in lines:
            anns = line.replace("\n", "")
            filename, attributes = anns.split("  ")
            attributes = attributes.split(" ")
            self.samples.append((filename, attributes))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fname, atts = self.samples[index]
        raise NotImplementedError


class SingleFolder(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        exts = {"bmp", "jpg", "jpeg", "pgm", "png",
                "ppm", "tif", "tiff", "webp"}
        files = os.listdir(path)
        files = [*filter(lambda x: x.split(".")[-1].lower() in exts, files)]
        self.filenames = [*map(lambda f: os.path.join(path, f), files)]
        self.transform = transform
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


class StarGANv2Dataset(datasets.ImageFolder):
    def __init__(self, root, train=True, transform=None):
        dataset = root.split("/")[-1]
        assert os.path.exists(root), \
            f"{dataset} dataset does not exists!. You can download" \
            " it from 'https://github.com/clovaai/stargan-v2'."
        split = "train" if train else "split"
        super().__init__(os.path.join(root, split), transform)
