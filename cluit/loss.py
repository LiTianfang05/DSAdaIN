import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.torch_utils import concat_all_gather


def hinge_adversarial_loss(phase, logit_r=None, logit_f=None):
    if phase == "D":
        loss_r = F.relu(1.0 - logit_r).mean()
        loss_f = F.relu(1.0 + logit_f).mean()
        loss = loss_r + loss_f
    else:
        loss = -logit_f.mean()
    return loss


def nonsat_adversarial_loss(phase, logit_r=None, logit_f=None):
    if phase == "D":
        loss_r = F.softplus(-logit_r).mean()
        loss_f = F.softplus(logit_f).mean()
        loss = loss_r + loss_f
    else:
        loss = F.softplus(-logit_f).mean()
    return loss


def get_adversarial_loss(method):
    assert method in ("hinge", "nonsat")
    if method == "hinge":
        return hinge_adversarial_loss
    else:
        return nonsat_adversarial_loss


def compute_grad_gp(d_out, x_in, gamma=1.0, is_patch=False):
    r1_grads = torch.autograd.grad(
        outputs=d_out.sum() if not is_patch else d_out.mean(), inputs=x_in,
        create_graph=True, only_inputs=True,
    )[0]
    r1_penalty = r1_grads.square().sum((1, 2, 3))
    r1_penalty = r1_penalty * (gamma * 0.5)
    return (d_out * 0.0 + r1_penalty).mean()  # trick for DDP


class InfoNCELoss(nn.Module):
    def __init__(self, temperature, feature_dim, queue_size):
        super().__init__()
        self.tau = temperature
        self.queue_size = queue_size
        if torch.distributed.is_initialized():
            self.world_size = int(os.environ["WORLD_SIZE"])
        else:
            self.world_size = 1
        data = torch.randn(feature_dim, queue_size)
        data = F.normalize(data, dim=0)
        self.register_buffer("queue_data", data)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, query, key):
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", (query, key)).unsqueeze(-1)
        # negative logits: NxK
        queue = self.queue_data.clone().detach()
        l_neg = torch.einsum("nc,ck->nk", (query, queue))
        # logits: Nx(1+K)
        logits = torch.cat((l_pos, l_neg), dim=1)
        # labels: positive key indicators
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=query.device)
        return F.cross_entropy(logits / self.tau, labels)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        # gather from all gpus
        if self.world_size > 1:
            keys = concat_all_gather(keys, self.world_size)
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_data[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size
