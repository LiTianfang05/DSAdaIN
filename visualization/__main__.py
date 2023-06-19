#!/usr/bin/env python
import argparse
import importlib
import os
import torch
import visualization
from core.data import get_dataset
from core import misc, torch_utils
from cluit import CLUIT
from cluit.augmentation import TestTransform


def parse_args():
    parser = argparse.ArgumentParser()
    for k, v in visualization.__dict__.items():
        if "Visualizer" in k:
            parser = v.add_commandline_args(parser)
    parser.add_argument("--tasks", type=str, nargs="+", default=['swap'])
    parser.add_argument("--checkpoint", type=str, default='E:/计算机科学论文/实验结果/对比实验/our/checkpoints/200k_checkpoint.pt')
    parser.add_argument("--folder", type=str, default='E:/计算机科学论文/实验结果/对比实验/our/afhq/wild')
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--run-dir", type=str, default="run")
    parser.add_argument("--cudnn-bench", type=misc.arg2bool, default=True)
    parser.add_argument("--allow-tf32", type=misc.arg2bool, default=False)
    return parser.parse_args()


def find_visualizer_using_name(visualizer_name):
    visualizer_filename = "visualization.{}_visualizer".format(visualizer_name)
    visualizerlib = importlib.import_module(visualizer_filename)
    # In the file, the class called LauncherNameLauncher() will
    # be instantiated. It has to be a subclass of BaseLauncher,
    # and it is case-insensitive.
    visualizer = None
    visualizer_name = visualizer_name.replace("_", "")
    for name, cls in visualizerlib.__dict__.items():
        if name.lower() == visualizer_name + "visualizer":
            visualizer = cls
    if visualizer is None:
        raise ValueError("In %s.py, there should be a class named Visualizer")
    return visualizer


def main():
    args = parse_args()
    assert os.path.isfile(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    opt = checkpoint["opt"]
    args.image_size = opt.image_size
    args.run_dir = opt.run_dir
    print(opt)
    print(f"=> cuDNN benchmark = {opt.cudnn_bench}")
    torch.backends.cudnn.benchmark = opt.cudnn_bench
    print(f"=> allow tf32 = {opt.allow_tf32}")
    torch.backends.cuda.matmul.allow_tf32 = opt.allow_tf32
    torch.backends.cudnn.allow_tf32 = opt.allow_tf32
    print(f"=> random seed = {opt.seed}")
    torch_utils.set_seed(opt.seed)
    model = CLUIT(opt)
    model.load(checkpoint)
    del model.optimizers
    torch.cuda.empty_cache()
    transform = TestTransform(opt.image_size)
    dataset = get_dataset(args.folder, transform)
    for task in args.tasks:
        # visualizer = find_visualizer_using_name(task)(**vars(opt))
        visualizer = find_visualizer_using_name(task)(**vars(args))
        if visualizer.is_possible():
            visualizer.visualize(model, dataset)


if __name__ == "__main__":
    main()
