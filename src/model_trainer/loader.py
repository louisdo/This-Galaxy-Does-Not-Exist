"""This file contains the dataloader"""
import cv2, torch, os
import scipy.io
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from astroNN.datasets import galaxy10
from torchvision import transforms
from PIL import Image


class Galaxy10Dataset(Dataset):
    def __init__(self, imsize: int = 64) -> None:
        # This will download the data at the first time being run
        images, labels = galaxy10.load_data()

        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int)

        # The mean and std is from imagenet
        mean = [123.675, 116.28 , 103.53]
        std = [58.395, 57.12 , 57.375]
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std),
                                             transforms.Resize(imsize)])
        

    def __len__(self) -> int:
        return len(self.images)


    def __getitem__(self, index) -> ("torch.tensor", "torch.tensor"):
        img = self.images[index]
        labels = np.array(self.labels[index])
        return self.transform(img), torch.from_numpy(labels)



class StanfordCarDataset(Dataset):
    def __init__(self, data_folder, labels_file, imsize):
        im_files = os.listdir(data_folder)
        im_files = [os.path.join(data_folder, f) for f in im_files]
        self.im_files = im_files

        mat_file = scipy.io.loadmat(labels_file)["annotations"][0]
        self.labels = {os.path.join(data_folder, mat_file[index][-1]) : mat_file[index][-2] for index in range(len(mat_file))}

        # The mean and std is from imagenet
        mean = [123.675, 116.28 , 103.53]
        std = [58.395, 57.12 , 57.375]
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std),
                                             transforms.Resize(imsize)])

    def load_image(self, im_file):
        return Image.open(im_file)

    def generate_one_hot_label(self, label):
        res = np.zeros([196])
        res[label - 1] = 1
        return res.astype(np.int)

    def __len__(self) -> int:
        return len(self.im_files)

    def __getitem__(self, index) -> ("torch.tensor", "torch.tensor"):
        img = self.load_image(self.im_files[index])
        label = self.generate_one_hot_label(self.labels[self.im_files[index]])
        return self.transform(img), torch.from_numpy(label)