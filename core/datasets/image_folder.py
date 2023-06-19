import os
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = ('jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp')


def is_image_file(file):
    return file.split(".")[-1].lower() in IMG_EXTENSIONS


def make_dataset_from_folder(dir, max_dataset_size=float("inf")):
    assert os.path.isdir(dir), f"{str(dir)} is not a valid directory."
    samples = []
    label, target = None, 0
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                if label is None:
                    label = root
                elif root != label:
                    label = root
                    target += 1
                path = os.path.join(root, fname)
                samples.append((path, target))
    return samples[:min(max_dataset_size, len(samples))]


class ImageFolder(Dataset):
    def __init__(self, root, transform=None, return_target=False):
        self.samples = make_dataset_from_folder(root)
        if not self.samples:
            raise RuntimeError(
                f"Found 0 images in: {root}\n"
                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS))
        self.classes = set([y for _, y in self.samples])
        self.transform = transform
        self.return_target = return_target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return (image, target) if self.return_target else image
