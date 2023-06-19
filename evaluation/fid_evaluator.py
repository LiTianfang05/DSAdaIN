import os
import shutil

import cleanfid
import torch
import torchvision.transforms.functional as VF
from cleanfid import fid
from PIL import Image

from core.datasets.image_folder import is_image_file
from .base_evaluator import BaseEvaluator


class FIDEvaluator(BaseEvaluator):

    @staticmethod
    def add_commandline_args(parser):
        parser.add_argument(
            "--eval_freq", default=50_000, type=int,
            help="Number of samples to generate for FID/KID."
        )
        return parser

    def __init__(self, eval_freq, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = 5 * eval_freq

    def prepare_evaluation(self):
        print("Preparing FID evaluation ...")
        
        if self.test_dataset.endswith("/"):
            self.test_dataset = self.test_dataset[:-1]
        if "afhq" in self.test_dataset:
            dataset = "afhq"
        elif "celeba_hq" in self.test_dataset:
            dataset = "celeba_hq"
        elif "church" in self.test_dataset:
            dataset = "lsun_church"
        elif "ffhq" in self.test_dataset:
            dataset = "ffhq"
        else:
            raise ValueError
        print(f"\tDataset: {dataset}")

        self.clean_fid_kwargs = {
            "mode": "clean",
            "dataset_res": self.image_size,
        }
        if dataset == "lsun_church" and self.image_size == 256:
            self.clean_fid_kwargs["dataset_name"] = dataset
            self.clean_fid_kwargs["dataset_split"] = "train"
        elif dataset == "ffhq" and self.image_size in (256, 1024):
            self.clean_fid_kwargs["dataset_name"] = dataset
            self.clean_fid_kwargs["dataset_split"] = "trainval70k"
        else:
            dataset_name = f"{dataset}{self.image_size}"
            self.clean_fid_kwargs["dataset_name"] = dataset_name
            self.clean_fid_kwargs["dataset_split"] = "custom"
            if not fid.test_stats_exists(dataset_name, mode="clean"):
                print(f"Creating custom stats '{dataset_name}'...")
                self.make_custom_stats(self.train_dataset, dataset_name)
                print("Done!\n")

        print(f"\tDevice: {self.device}")
        print("Done!")
        self._is_possible = True

    @torch.no_grad()
    def evaluate(self, model, dataset=None, batch_size=50, *args, **kwargs):
        print("Evaluate FID ...")

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True, num_workers=0, drop_last=True
        )

        path_fake = os.path.join(self.run_dir, "fid")
        shutil.rmtree(path_fake, ignore_errors=True)
        os.makedirs(path_fake)
        
        num_fake = 0
        print(f'Generating images ..')
        print(self.num_samples)
        while num_fake < self.num_samples:
            for x_src in loader:

                x_src = x_src.to(self.device)
                s_ref = model.prototypes_ema.sample(x_src.size(0))
                # s_ref = model.D_ema(x_src, command="encode")[torch.randperm(x_src.size(0))]

                x_fake = model.forward(x_src, s_ref)
                x_fake = x_fake.mul(127.5).add(128).clamp(0, 255)
                x_fake = x_fake.to(torch.uint8)

                # Save generated images to calculate FID later.
                for i in range(x_src.size(0)):
                    num_fake += 1
                    filename = os.path.join(path_fake, f'{num_fake:05d}.png')
                    pil_image = VF.to_pil_image(x_fake[i])
                    pil_image.save(filename)

                    if num_fake >= self.num_samples:
                        break
                if num_fake >= self.num_samples:
                    break

        del loader
        torch.cuda.empty_cache()
        fid_value = fid.compute_fid(path_fake, **self.clean_fid_kwargs)
        # kid_value = fid.compute_kid(path_fake, **self.clean_fid_kwargs) * 1000
        # return {"FID": fid_value, "KID": kid_value}
        return {"FID": fid_value}

    def make_custom_stats(self, path, dataset_name):
        tmpdir = "/tmp/fid_cache"
        while os.path.exists(tmpdir):
            tmpdir += "a"
        os.makedirs(tmpdir)

        n = 0
        for root, _, fnames in sorted(os.walk(path, followlinks=True)):
            for fname in fnames:
                if not is_image_file(fname):
                    continue
                image = Image.open(os.path.join(root, fname)).convert("RGB")
                image = image.resize((self.image_size,)*2, resample=Image.BICUBIC)
                image.save(os.path.join(tmpdir, f"{n}.png"))
                n += 1
        
        stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
        os.makedirs(stats_folder, exist_ok=True)
        fid.make_custom_stats(dataset_name, tmpdir, mode="clean")
        shutil.rmtree(tmpdir)