import torch, torchvision, sys
sys.path.append("../..")
import numpy as np
from generator import Generator
from argparse import ArgumentParser
from utils.utils import Utils
from PIL import Image


class Infer:
    def __init__(self, CONFIG: dict, device: str = "cpu"):
        self.CONFIG = CONFIG
        self.generator = Generator(latent_dim = self.CONFIG["train"]["latent_dim"],
                                   number_channel = self.CONFIG["train"]["number_channel"],
                                   image_size = self.CONFIG["train"]["image_size"])
        self.generator = self.generator.to(device)

        self.device = torch.device(device)
    
    def load_weight(self, ckpt_path):
        self.generator.load_state_dict(torch.load(ckpt_path))

    def __call__(self, num_images):
        noise = torch.randn(num_images, self.CONFIG["train"]["latent_dim"], 1, 1).to(self.device)
        with torch.no_grad():
            generated = self.generator(noise).detach().cpu()
            #generated = torchvision.utils.make_grid(generated, padding=2, normalize=True)

        res = torchvision.utils.make_grid(generated, padding = 2, normalize = True).cpu()
        return res


if __name__ == "__main__":
    parser = ArgumentParser(description = "Model trainer")
    parser.add_argument("--ckpt_path", help = "Path to model checkpoint", required = True)
    parser.add_argument("--num_images", help = "Number of images to infer", type = int, required = True)
    parser.add_argument("--where_to", help = "Where to save the inference result", required = True)
    args = parser.parse_args()

    CONFIG = Utils.get_config_yaml("../../config/config.yml")
    infer = Infer(CONFIG = CONFIG, device = "cuda")
    infer.load_weight(args.ckpt_path)

    infer_res = infer(args.num_images)
    infer_res = (infer_res.permute(1, 2, 0) * 255).int()
    infer_res = Image.fromarray(infer_res.data.numpy().astype(np.uint8))

    infer_res.save(args.where_to)