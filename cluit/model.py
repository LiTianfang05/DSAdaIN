import copy
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
from cluit import loss, net
from core import BaseModel
from core.torch_utils import unnormalize, update_average
from cluit import unet
import clip


class CLUIT(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.lerps = None
        self.debug = None
        self.x_fake = None
        self.do_lazy_r1 = None
        self.x_ref = None
        self.batch_size = None
        self.x_aug = None
        self.x_real = None

    @staticmethod
    def add_commandline_args(parser):
        # Model parameters.
        parser.add_argument("--style-dim", default=128, type=int)
        parser.add_argument("--hypersphere-dim", default=256, type=int)
        parser.add_argument("--queue-size", default=2048, type=int)
        # Hyperparameters.
        parser.add_argument("--temperature", default=0.07, type=float)
        parser.add_argument("--gamma", default=10.0, type=float)
        parser.add_argument("--lambda-cyc", default=3.0, type=float)
        parser.add_argument("--lambda-Dcont", default=0.1, type=float)
        parser.add_argument("--lambda-Gcont", default=7.0, type=float)
        # Optimizer parameters.
        parser.add_argument("--lr", default=5e-5, type=float)
        parser.add_argument("--beta1", default=0.0, type=float)
        parser.add_argument("--beta2", default=0.99, type=float)
        parser.add_argument("--lazy-r1-freq", default=16, type=int)
        return parser

    def _create_networks(self):
        opt = self.opt
        self.G = unet.UNET(opt.style_dim).cuda()
        self.E = net.StyleEncoder(opt.image_size, opt.style_dim).cuda()
        self.D = net.Discriminator(opt.image_size, opt.hypersphere_dim).cuda()
        self.G_ema = copy.deepcopy(self.G).requires_grad_(False)
        self.D_ema = copy.deepcopy(self.D).requires_grad_(False)
        self.E_ema = copy.deepcopy(self.E).requires_grad_(False)

    def _create_criterions(self):
        opt = self.opt
        self.gan_loss = loss.nonsat_adversarial_loss
        self.nce_loss = loss.InfoNCELoss(opt.temperature, opt.hypersphere_dim, opt.queue_size).cuda()
        self.model_clip, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opt.style_dim // 32)

    def _create_optimizers(self):
        opt = self.opt
        self.optimizers = {"G": torch.optim.Adam(
            [
                {"params": self.E.parameters()},
                {"params": self.G.parameters()}
            ],
            lr=opt.lr, betas=(opt.beta1, opt.beta2))}
        c = opt.lazy_r1_freq / (1. + opt.lazy_r1_freq)
        self.optimizers["D"] = torch.optim.Adam(
            [
                {"params": self.D.parameters()}
            ],
            lr=opt.lr, betas=(opt.beta1 ** c, opt.beta2 ** c))

    def forward(self, input, reference):
        style_code = self.E(reference)
        x_fake = self.G(input, style_code)
        return x_fake

    def set_input(self, step, xs):
        self.step = step
        self.x_real, self.x_aug = xs
        self.batch_size = self.x_real.size(0)
        self.x_ref = self.x_real[torch.randperm(self.batch_size)]
        self.do_lazy_r1 = step % self.opt.lazy_r1_freq == 0 and self.opt.gamma > 0.0

    def training_step(self):
        self.loss.clear()
        self.x_fake = self.forward(self.x_real, self.x_ref)
        self._update_discriminator()
        self._update_generator()
        self._update_average()
        self.nce_loss.dequeue_and_enqueue(self.key)
        return self.loss

    def CLIPLoss(self, image1, image2):
        query = self.model_clip.encode_image(self.avg_pool(self.upsample(image1)))
        key = self.model_clip.encode_image(self.avg_pool(self.upsample(image2)))
        sim = nn.CosineSimilarity()
        loss_clip = (1.0-sim(query, key)).mean()
        return loss_clip

    def _update_discriminator(self):
        self.D.requires_grad_(True)
        self.optimizers["D"].zero_grad(set_to_none=True)
        logit_r, _ = self.D(self.x_ref, project=False)
        logit_f, _ = self.D(self.x_fake.detach(), project=False)
        # Compute the adversarial loss.
        loss_adv = self.gan_loss("D", logit_r, logit_f)
        # Compute the match contrastive loss.
        _, query = self.D(self.x_real, logit=False)
        with torch.no_grad():
            _, key = self.D_ema(self.x_aug, logit=False)
            key = key.detach()
        loss_cont = self.nce_loss(query, key)
        loss_cont = self.opt.lambda_Dcont * loss_cont
        # Update the discriminator parameters.
        d_loss = loss_adv + loss_cont
        d_loss.backward()
        self.optimizers["D"].step()
        self.loss["D/cont"] = loss_cont.detach()
        self.loss["D/adv"] = loss_adv.detach()
        self.key = key
        # Compute the R1 gradient penalty.
        if self.do_lazy_r1:
            self.optimizers["D"].zero_grad(set_to_none=True)
            self.x_ref.requires_grad_(True)
            logit, _ = self.D(self.x_ref, project=False)
            r1 = loss.compute_grad_gp(logit, self.x_ref, self.opt.gamma)
            lazy_r1 = self.opt.lazy_r1_freq * r1
            self.x_ref.requires_grad_(False)
            lazy_r1.backward()
            self.x_ref.grad = None
            self.optimizers["D"].step()
            self.loss["D/R1"] = lazy_r1.detach()

    def _update_generator(self):
        self.D.requires_grad_(False)
        self.optimizers["G"].zero_grad(set_to_none=True)
        # Calculate the adversarial loss.
        logit, _ = self.D(self.x_fake, project=False)
        loss_adv = self.gan_loss("G", logit_f=logit)
        # Calculate the cycle-consistency loss.
        x_recon = self.forward(self.x_fake, self.x_real)
        loss_cyc = F.l1_loss(x_recon, self.x_real).cuda()
        # Calculate the style contrastive loss.
        loss_cont = self.CLIPLoss(self.x_fake, self.x_ref)
        loss_cyc = self.opt.lambda_cyc * loss_cyc
        loss_cont = self.opt.lambda_Gcont * loss_cont
        # Update the generator and style encoder parameters.
        g_loss = loss_adv + loss_cont + loss_cyc
        g_loss.backward()
        self.optimizers["G"].step()
        self.loss["G/adv"] = loss_adv.detach()
        self.loss["G/cyc"] = loss_cyc.detach()
        self.loss["G/cont"] = loss_cont.detach()

    def _update_average(self):
        update_average(self.G, self.G_ema)
        update_average(self.D, self.D_ema)
        update_average(self.E, self.E_ema)

    def prepare_snapshot(self, dataset):
        indices = random.sample([i for i in range(len(dataset))], k=6)
        debug = []
        for i in indices[:4]:
            x = dataset[i]
            debug.append(x)
        self.debug = torch.stack(debug).to(self.device)
        self.lerps = []
        for i in indices[4:]:
            x = dataset[i]
            x = x.unsqueeze(0)
            self.lerps.append(x.to(self.device))

    def snapshot(self):
        opt = self.opt
        num_debug = self.debug.size(0)
        snapshot_dir = os.path.join(opt.run_dir, f"snapshot/{self.step // 1000:03d}k")
        os.makedirs(snapshot_dir, exist_ok=True)
        # Reference.
        reference = [
            torch.ones_like(self.debug[0]).unsqueeze(0),
            self.debug]
        for i in range(num_debug):
            src = self.debug[i].unsqueeze(0)
            translated = self.forward(src.expand_as(self.debug), self.debug)
            reference.append(src)
            reference.append(translated)
        reference = unnormalize(torch.cat(reference))
        fname = os.path.join(snapshot_dir, "ref.png")
        save_image(reference, fname, nrow=num_debug + 1, padding=2)
