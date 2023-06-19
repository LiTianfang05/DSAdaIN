import glob
import json
import os
import torch


class BaseModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        assert hasattr(opt, "run_dir")
        self.opt = opt
        self.device = "cpu"
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.step = 0
        self.ckpt = os.path.join(opt.run_dir, "checkpoints")
        self.loss = {}
        self._create_networks()
        self._create_criterions()
        self._configure_gpu()
        self._create_optimizers()

    def _create_networks(self):
        raise NotImplementedError

    def _create_criterions(self):
        self.optimizers = {}

    def _configure_gpu(self):
        self.device = torch.device(self.rank)
        self.to(self.device)
        if self.world_size > 1:
            for name, module in self.named_children():
                if self.count_parameter(module) > 0:
                    module = torch.nn.parallel.DistributedDataParallel(
                        module,
                        device_ids=[self.rank],
                        broadcast_buffers=False,
                        find_unused_parameters=True)
                    setattr(self, name, module)

    def _create_optimizers(self):
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        raise NotImplementedError

    def save_options(self):
        with open(os.path.join(self.opt.run_dir, "options.json"), "w") as f:
            json.dump(vars(self.opt), f, indent=4)

    def log_parameters(self):
        for n, m in self.named_children():
            num_params = self.count_parameter(m)
            if num_params > 0:
                print(f"[{n}] has {num_params} trainable parameters.")
                setattr(self.opt, f"{n}_param", num_params)

    def save(self, step, **kwargs):
        state = {"opt": self.opt, "step": step}
        state.update(**kwargs)
        for name, net in self.named_children():
            if hasattr(net, "module"):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()
            state[name+"_state_dict"] = state_dict
        for name, optim in self.optimizers.items():
            state[name+"_optimizer"] = optim.state_dict()
        ckpt = self.ckpt
        os.makedirs(ckpt, exist_ok=True)
        ckpt = ckpt + '/{}_checkpoint.pt'.format(f"{step//1000:03d}k")
        torch.save(state, ckpt)
        print(f"=> Saved '{ckpt}'")

    def load(self, checkpoint=None):
        if isinstance(checkpoint, str):
            assert os.path.isfile(checkpoint), \
                checkpoint + "is not a valid checkpoint file."
            print(f"=> Loading checkpoint '{checkpoint}'")
            checkpoint = torch.load(checkpoint, map_location="cpu")
        elif checkpoint is None:
            checkpoints = os.path.join(self.opt.run_dir, "checkpoints/*.pt")
            checkpoints = sorted(glob.glob(checkpoints))
            if not checkpoints:
                print("Found 0 checkpoints.")
                return
            checkpoint = checkpoints[-1]
            print(f"=> Loading checkpoint '{checkpoint}'")
            checkpoint = torch.load(checkpoint, map_location="cpu")
        print(f"\tstep={checkpoint['step']}")
        self.step = checkpoint["step"]
        for name, net in self.named_children():
            key = name + "_state_dict"
            if key in checkpoint.keys():
                state_dict = checkpoint[key]
                if hasattr(net, "module"): 
                    net.module.load_state_dict(state_dict)
                else:
                    net.load_state_dict(state_dict)
                print(f"\tLoaded state dict from {key}")
            else:
                print(f"\tFailed to load {key}")
        for name, opt in self.optimizers.items():
            key = name + "_optimizer"
            if key in checkpoint.keys():
                opt.load_state_dict(checkpoint[key])
                print(f"\tLoaded state dict from {key}")
            else:
                print(f"\tFailed to load {key}")
        print(f"=> Done!")

    @staticmethod
    def count_parameter(net):
        num_params = 0
        for p in net.parameters():
            if p.requires_grad:
                num_params += p.numel()
        return num_params
