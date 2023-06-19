import torch.nn as nn
from PIL import Image
from torchvision import transforms


class Augmentation(nn.Module):
    @staticmethod
    def add_commandline_args(parser):
        parser.add_argument("--min-scale-crop", default=0.125, type=float)
        parser.add_argument("--rotation-degree", default=20.0, type=float)
        parser.add_argument("--color-jitter-scale", default=0.5, type=float)
        return parser

    def __init__(self, image_size, min_scale_crop, rotation_degree, color_jitter_scale, mean=(0.5, 0.5, 0.5),
                 std=(0.5, 0.5, 0.5), **kwargs):
        super().__init__()
        self.transform = []
        crop = transforms.Compose([
            transforms.Resize((int(image_size * 1.25),)*2),
            transforms.RandomCrop(image_size)])
        resize = transforms.Resize((image_size,)*2)
        resize_fn = transforms.RandomChoice([crop, resize])
        self.transform.append(
            transforms.Compose([
                resize_fn,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]))
        self.transform.append(
            transforms.Compose([
                transforms.RandomAffine(
                    rotation_degree, translate=(0.1, 0.1), shear=10),
                transforms.RandomResizedCrop(
                    size=image_size,
                    scale=(min_scale_crop, 1.0)),
                transforms.RandomHorizontalFlip(),
                get_color_distortion(color_jitter_scale),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)]))
        
    def __call__(self, x):
        return list(map(lambda T: T(x), self.transform))


class TestTransform(transforms.Compose):
    def __init__(self, image_size, resample=Image.BICUBIC, **kwargs):
        super().__init__([
            transforms.Resize((image_size,)*2, resample),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_color_distortion(s):
    jiter_params = tuple(map(lambda x: x * s, (0.8, 0.8, 0.8, 0.2)))
    color_jitter = transforms.ColorJitter(*jiter_params)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    # rnd_gray = transforms.RandomGrayscale(p=0.2)
    # color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    # return color_distort
    return rnd_color_jitter
