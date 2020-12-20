import json, torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from inception import InceptionV3
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from scipy import linalg
from model_trainer.loader import Galaxy10Dataset


class EvalDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        assert "path" in self.data.columns

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_shape = 299

        self.transform = transforms.Compose([transforms.Resize(int(input_shape*1.15)),
                                            transforms.CenterCrop(input_shape),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)])


    @staticmethod
    def load_image(img_path: str, mode: str = "np") -> "np.array or PIL.Image":
        """
        input:
            + img_path: path to image
            + mode: should be either 'PIL' or 'np'
        output:
            + if mode is 'np': return a numpy array image
            + if mode is 'PIL': return a pillow image
        """
        def _raise_image_error(img: "Pillow image"):
            num_channels = len(img.getbands())
            assert num_channels == 3, "RGB image should have 3 channels instead of {}".format(num_channels)

        pil_image = Image.open(img_path).convert("RGB")
        _raise_image_error(pil_image)

        if mode == "PIL": return pil_image
        else: return np.array(pil_image)


    def __getitem__(self, index):
        img_path = self.data.path.loc[index]
        img = self.load_image(img_path, mode = "PIL")
        transformed_img = self.transform(img)
        return transformed_img



class Evaluation:
    def __init__(self, 
                 fake_data_path: str,
                 device: torch.device):
        real_dataset = Galaxy10Dataset(imsize = 299)
        fake_dataset = EvalDataset(data_path = fake_data_path)

        self.real_dataloader = DataLoader(real_dataset,
                                          batch_size = 8,
                                          shuffle = False,
                                          num_workers = 2,
                                          drop_last = False)
        self.fake_dataloader = DataLoader(fake_dataset,
                                          batch_size = 8,
                                          shuffle = False,
                                          num_workers = 2,
                                          drop_last = False)

        self.device = device
        self.dims = 2048

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = InceptionV3([block_idx]).to(device)


    def infer_features(self, which: str):
        assert which in ["real", "fake"]

        loaders = {
            "real": self.real_dataloader,
            "fake": self.fake_dataloader
        }

        loader = loaders[which]
        pred_arr = np.empty((len(loader.dataset), self.dims))

        train_pbar = tqdm(loader, desc = f"Evaluation for {which} data")
        for batch_idx, batch in enumerate(train_pbar):
            batch = batch.to(self.device)

            with torch.no_grad():
                pred = self.model(batch)[0]

            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[batch_idx:batch_idx + pred.shape[0]] = pred

        return pred_arr

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)


    def fid_score(self):
        real_features = self.infer_features(which = "real")
        fake_features = self.infer_features(which = "fake")

        mean_real = np.mean(real_features, axis = 0)
        mean_fake = np.mean(fake_features, axis = 0)

        cov_real = np.cov(real_features, rowvar = False)
        cov_fake = np.cov(fake_features, rowvar = False)

        score = self.calculate_frechet_distance(mean_real, cov_real, mean_fake, cov_fake)

        return score