#!/usr/bin/env python
import argparse
import importlib
import os

import torch

import evaluation
from core import misc, torch_utils
from core.data import get_dataset
from cluit import CLUIT
from cluit.augmentation import TestTransform


def parse_args():
    parser = argparse.ArgumentParser()
    for k, v in evaluation.__dict__.items():
        if "Evaluator" in k:
            parser = v.add_commandline_args(parser)
    parser.add_argument("tasks", type=str, nargs="+")
    parser.add_argument("--checkpoint", type=str, default='/xxxy3408hppc/xxxy3408_24/ltf/our/checkpoints/200k_checkpoint.pt')
    parser.add_argument("--batch-size", default=8, type=int)
    
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=str, default="runs")

    parser.add_argument("--train-dataset", type=str, default='/xxxy3408hppc/xxxy3408_24/ltf/our/datasets/afhq/train')
    parser.add_argument("--test-dataset", type=str, default='/xxxy3408hppc/xxxy3408_24/ltf/our/datasets/afhq/val')

    parser.add_argument("--cudnn-bench", type=misc.arg2bool, default=True)
    parser.add_argument("--allow-tf32", type=misc.arg2bool, default=False)
    return parser.parse_args()


def find_evaluator_using_name(evaluator_name):
    evaluator_filename = "evaluation.{}_evaluator".format(evaluator_name)
    evaluatorlib = importlib.import_module(evaluator_filename)

    # In the file, the class called LauncherNameLauncher() will
    # be instantiated. It has to be a subclass of BaseLauncher,
    # and it is case-insensitive.
    evaluator = None
    evaluator_name = evaluator_name.replace("_", "")
    for name, cls in evaluatorlib.__dict__.items():
        if name.lower() == evaluator_name + "evaluator":
            evaluator = cls

    if evaluator is None:
        raise ValueError("In %s.py, there should be a class named Evaluator")

    return evaluator


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
    del model.optimizers  # NOTE: we don't need optimizers here.
    torch.cuda.empty_cache()

    # Override options.
    for k, v in vars(args).items():
        if hasattr(opt, k) and v is not None:
            setattr(opt, k, v)

    transform = TestTransform(opt.image_size)
    eval_dataset = get_dataset(opt.test_dataset, transform)

    results = {}
    for task in args.tasks:
        evaluator = find_evaluator_using_name(task)(**vars(opt))
        if evaluator.is_possible():
            result = evaluator.evaluate(model, eval_dataset, step=model.step)
            results.update(result)
    print(results)
    misc.report_metric(results, run_dir=opt.run_dir)


if __name__ == "__main__":
    main()