import glob
import os
import torch
from PIL import Image
from torchvision import transforms
from .base_visualizer import BaseVisualizer
from thop import profile


class SwapVisualizer(BaseVisualizer):
    @staticmethod
    def add_commandline_args(parser):
        parser.add_argument("--style-folder", type=str, default='E:/计算机科学论文/实验结果/对比实验/our/afhq/cat')
        return parser

    def __init__(self, style_folder, **kwargs):
        self.fname = None
        self.fnames = None
        self.fnamec = None
        self.fnameo = None
        self._is_possible = None
        self.transform = None
        self.style_folder = style_folder
        # super().__init__(**kwargs)
        super().__init__(**kwargs)
        self.content_folder = self.folder

    def prepare_visualization(self):
        print("Preparing swapping visualization ...")
        if self.folder is None or self.style_folder is None:
            print("\tcontent/style folder is not provided.")
            return
        result_dir = os.path.join(self.run_dir, "swap")
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(result_dir + '/c/', exist_ok=True)
        os.makedirs(result_dir + '/s/', exist_ok=True)
        os.makedirs(result_dir + '/t/', exist_ok=True)
        self.fname = os.path.join(result_dir, "all_{}.png")
        self.fnamec = os.path.join(result_dir + '/c/', "c_{}_{}.png")
        self.fnames = os.path.join(result_dir + '/s/', "s_{}_{}.png")
        self.fnameo = os.path.join(result_dir + '/t/', "t_{}_{}.png")
        self.transform = transforms.Compose([
            # transforms.Resize((self.image_size,)*2, Image.BICUBIC),
            transforms.Resize(self.image_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
        print(f"\tDevice: {self.device}")
        print("Done!")
        self._is_possible = True

    @torch.no_grad()
    def visualize(self, model, *args, **kwargs):
        print("Visualize swap synthesis ...")
        content_images = []
        for content_image in glob.glob(os.path.join(self.folder, "*.*")):
            content_image = Image.open(content_image).convert("RGB")
            content_image = self.transform(content_image)
            content_images.append(content_image.unsqueeze(0).to(self.device))
        grid = [torch.ones_like(content_images[0])]
        style_images = []
        for fname in glob.glob(os.path.join(self.style_folder, "*.*")):
            image = Image.open(fname).convert("RGB")
            image = image.resize((self.image_size,) * 2, Image.BICUBIC)
            image = self.transform(image)
            style_images.append(image)
        style_images = torch.stack(style_images).to(self.device)
        grid.append(style_images)
        # for content_image in content_images:
        for i, content_image in enumerate(content_images):
            # inputs = content_image.expand_as(style_images)
            inputs = content_image.repeat(style_images.size(0), 1, 1, 1)
            outputs = model.forward(inputs, style_images)
            flops1, params1 = profile(model, inputs=(inputs, style_images))
            params2 = 20461185
            flops = flops1
            params = params1 + params2
            print('the model FLOPs is {}G'.format(round(flops / (10 ** 9), 2)))
            print('the model Params is {}M'.format(round(params / (10 ** 6), 2)))
            for j, output in enumerate(outputs):
                self.save_image(content_image, self.fnamec.format(j + 1, i + 1))
                self.save_image(style_images[i], self.fnames.format(i + 1, j + 1))
                self.save_image(output, self.fnameo.format(j + 1, i + 1))
            grid.append(content_image)
            grid.append(outputs)
        grid = torch.cat(grid)
        nrow = style_images.size(0) + 1
        self.save_image(grid, self.fname.format("reference"), nrow=nrow)
        # for i, im in enumerate(grid):
        #     self.save_image(im, self.fname.format(i), nrow=nrow)
