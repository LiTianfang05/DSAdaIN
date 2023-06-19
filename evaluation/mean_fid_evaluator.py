import os
import shutil

import torch
import torchvision.transforms.functional as VF
from cleanfid import fid

from .fid_evaluator import FIDEvaluator


class MeanFIDEvaluator(FIDEvaluator):

    @staticmethod
    def add_commandline_args(parser):
        parser.add_argument("--gamma", default=10, type=int, help="mFID")
        parser.add_argument("--eval-kid", action="store_true")
        return parser

    def __init__(self, gamma, eval_kid=False, **kwargs):
        super().__init__(**kwargs)
        self.num_repeat = gamma
        self.eval_kid = eval_kid

    def prepare_evaluation(self):
        print("Preparing mFID evaluation ...")
        
        if self.test_dataset.endswith("/"):
            self.test_dataset = self.test_dataset[:-1]
        splits = self.test_dataset.split("/")
        if splits[-1] in ("eval", "test", "train", "val", "valid"):
            dataset = splits[-2]
        else:
            dataset = splits[-1]
        print(f"\tDataset: {dataset}")

        domains = os.listdir(self.test_dataset)
        domains.sort()
        print(f"\tNumber of domains: {len(domains)}")
        if len(domains) < 2 or "data.mdb" in domains:
            print("\tmFID evaluation requires more than 2 domains.")
            self._is_possible = False
            return
        elif len(domains) > 5:
            print("\tmFID evaluation for many domains (>5) not implemented.")
            self._is_possible = False
            return

        self.tasks = []
        for ref_domain in domains:
            ref_path = os.path.join(self.test_dataset, ref_domain)
            ref_dataset = self.get_eval_dataset(ref_path)
            dataset_name = f"{dataset}{self.image_size}_{ref_domain}"
            if not fid.test_stats_exists(dataset_name, mode="clean"):
                print(f"Creating custom stats '{dataset_name}'...")
                path_real = os.path.join(self.train_dataset, ref_domain)
                self.make_custom_stats(path_real, dataset_name)
                print("Done!\n")
            
            src_domains = [x for x in domains if x != ref_domain]
            for src_domain in src_domains:
                task = f"{src_domain}2{ref_domain}"
                src_path = os.path.join(self.test_dataset, src_domain)
                src_dataset = self.get_eval_dataset(src_path)

                self.tasks.append({
                    "name": task,
                    "source": src_dataset,
                    "reference": ref_dataset,
                    "dataset_name": dataset_name
                })
        print(f"\tNumber of tasks: {len(self.tasks)}")
        print(f"\tDevice: {self.device}")
        print("Done!")
        self._is_possible = True

    @torch.no_grad()
    def evaluate(self, model, dataset=None, batch_size=50, *args, **kwargs):
        print("Evaluate mFID ...")

        results_dict = {}
        for task in self.tasks:
            path_fake = os.path.join(self.run_dir, "mfid", task["name"])
            shutil.rmtree(path_fake, ignore_errors=True)
            os.makedirs(path_fake)

            src_loader = torch.utils.data.DataLoader(
                task["source"], batch_size=batch_size,
                shuffle=True, num_workers=0, drop_last=True
            )
            ref_loader = torch.utils.data.DataLoader(
                task["reference"], batch_size=batch_size,
                shuffle=True, num_workers=0, drop_last=True
            )
            # TODO: more flexible to data length.
            assert len(src_loader) == len(ref_loader)

            n = 0
            print(f'Generating images for {task["name"]}...')
            print(self.num_repeat)
            for _ in range(self.num_repeat):
                for x_src, x_ref in zip(src_loader, ref_loader):
                    x_src = x_src.to(self.device)
                    x_ref = x_ref.to(self.device)

                    x_fake = model.forward(x_src, x_ref)
                    x_fake = x_fake.mul(127.5).add(128).clamp(0, 255)
                    x_fake = x_fake.to(torch.uint8)

                    # Save generated images to calculate FID later.
                    for i in range(x_src.size(0)):
                        filename = os.path.join(path_fake, f'{n:04d}.png')
                        pil_image = VF.to_pil_image(x_fake[i])
                        pil_image.save(filename)
                        n += 1

            del src_loader
            del ref_loader
            results_dict["FID_"+task["name"]] = fid.compute_fid(
                path_fake, dataset_name=task["dataset_name"],
                mode="clean", dataset_split="custom"
            )
            if self.eval_kid:
                results_dict["KID_"+task["name"]] = fid.compute_kid(
                    path_fake, dataset_name=task["dataset_name"],
                    mode="clean", dataset_split="custom"
                )

        torch.cuda.empty_cache()

        sum_fid, sum_kid = 0.0, 0.0
        for k, v in results_dict.items():
            if "FID" in k:
                sum_fid += v
            elif "KID" in k:
                sum_kid += v
        results_dict["mFID"] = sum_fid / len(self.tasks)
        if self.eval_kid:
            results_dict["mKID"] = sum_kid / len(self.tasks)
        return results_dict