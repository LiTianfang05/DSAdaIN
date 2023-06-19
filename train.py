import argparse
import json
import os
import random
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from cluit import CLUIT
from cluit.augmentation import Augmentation, TestTransform
from core import data, misc, torch_utils
import evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="CLUIT training script")
    parser = Augmentation.add_commandline_args(parser)
    parser = CLUIT.add_commandline_args(parser)
    # for k, v in evaluation.__dict__.items():
    #     if "Evaluator" in k:
    #         parser = v.add_commandline_args(parser)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="./runs")
    parser.add_argument("--desc", type=str, nargs="+", default=[])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--length", type=misc.arg2int, default="1.6M")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--train-dataset", type=str, default="./datasets/afhq/train")
    parser.add_argument("--test-dataset", type=str, default="./datasets/afhq/val")
    # Logging.
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--snapshot-freq", type=int, default=1000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--save-freq", type=int, default=10000)
    # metrics
    parser.add_argument(
        "--fid-start-after", type=int, default=10000,
        help="FID will be evaluated after N steps (default: 10,000)")
    # Misc.
    parser.add_argument("--allow-tf32", type=misc.arg2bool, default=False)
    parser.add_argument("--cudnn-bench", type=misc.arg2bool, default=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--tqdm", type=misc.arg2bool, default=True)
    return parser.parse_args()


def options_from_args(args):
    if args.seed is None:
        args.seed = random.randint(0, 999)
    # Create output folder.
    assert os.path.isdir(args.train_dataset)  # TODO: support other formats.
    if "celeba_hq" in args.train_dataset:
        dataset = "celeba_hq"
    elif "ffhq" in args.train_dataset:
        dataset = "ffhq"
        if args.image_size != 1024:
            dataset += str(args.image_size)
    else:
        splits = "train", "test", "val", "eval", ""
        dataset = args.train_dataset.split("/")
        dataset = [x for x in dataset if x not in splits][-1]
        desc = dataset.replace("_train", "").replace("_lmdb", "")  # NOTE for LSUNs.
    for d in args.desc:
        desc += f"-{d}"
    prev_run_ids = []
    if os.path.isdir(args.out_dir):
        prev_run_ids = [int(x.split("-")[0]) for x in os.listdir(args.out_dir)]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(args.out_dir, f'{cur_run_id:03d}-{desc}')
    return args


def load_options(resume_dir):
    option_file = os.path.join(resume_dir, "options.json")
    assert os.path.isfile(option_file)
    with open(option_file, "r") as f:
        opt_dict = json.load(f)
    return argparse.Namespace(**opt_dict)


def train(opt, rank):
    model = CLUIT(opt)
    model.load()
    model.cuda()
    train_transform = Augmentation(**vars(opt))
    train_dataset = data.get_dataset(
        opt.train_dataset, train_transform,
        seed=opt.seed, stream=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True)
    data_stream = iter(train_loader)
    test_transform = TestTransform(opt.image_size)
    test_dataset = data.get_dataset(opt.test_dataset, test_transform)
    iters = opt.length // opt.batch_size
    pbar = range(model.step+1, iters+1)
    if rank == 0:
        model.log_parameters()
        model.save_options()
        writer = SummaryWriter(opt.run_dir)
        model.prepare_snapshot(test_dataset)
        if opt.tqdm:
            pbar = tqdm(pbar)
    for step in pbar:
        xs = next(data_stream)
        xs = tuple(map(lambda x: x.to(rank, non_blocking=True), xs))
        model.set_input(step, xs)
        loss = model.training_step()
        if rank != 0:
            continue
        if step % opt.log_freq == 0:
            for k, v in loss.items():
                writer.add_scalar(k, v, step)
        if step % opt.snapshot_freq == 0:
            model.snapshot()
        is_best = False
        if step % opt.eval_freq == 0:
            result_dict = {"step": f"{step//1000}k"}
            for k, v in result_dict.items():
                if k == "step":
                    continue
                writer.add_scalar("Eval/"+k, v, step)
            misc.report_metric(result_dict, opt.run_dir)
        if (step % opt.save_freq == 0) or (step == iters) or is_best:
            model.save(step, nimg=step * opt.batch_size)


def main():
    args = parse_args()
    if args.resume is None:
        opt = options_from_args(args)
    else:
        opt = load_options(args.resume)
        print(f"resume training '{args.resume}'")
    if "LOCAL_RANK" not in os.environ.keys():
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"=> set cuda device = {local_rank}")
    torch.cuda.set_device(local_rank)
    print(f"=> cuDNN benchmark = {opt.cudnn_bench}")
    torch.backends.cudnn.benchmark = opt.cudnn_bench
    print(f"=> allow tf32 = {opt.allow_tf32}")
    torch.backends.cuda.matmul.allow_tf32 = opt.allow_tf32
    torch.backends.cudnn.allow_tf32 = opt.allow_tf32
    print(f"=> random seed = {opt.seed}")
    torch_utils.set_seed(opt.seed)
    if world_size > 1:
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    if local_rank == 0:
        code_dir = os.path.join(opt.run_dir, "code")
        misc.copy_py_files(os.getcwd(), code_dir)
    train(opt, local_rank)


if __name__ == "__main__":
    main()
