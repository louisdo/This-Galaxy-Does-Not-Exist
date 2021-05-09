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
        mean = [0,0,0]
        std = [1,1,1]
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std),
                                             transforms.Resize(imsize)])
        

    def __len__(self) -> int:
        return len(self.images)


    def __getitem__(self, index) -> ("torch.tensor", "torch.tensor"):
        img = Image.open(self.images[index].astype(np.uint8))
        labels = np.array(self.labels[index])
        return self.transform(img), torch.from_numpy(labels)



class StanfordCarDataset(Dataset):
    def __init__(self, data_folder, labels_file, imsize):
        im_files = os.listdir(data_folder)
        im_files = [os.path.join(data_folder, f) for f in im_files]
        self.im_files = im_files

        mat_file = scipy.io.loadmat(labels_file)["annotations"][0]
        self.labels = {os.path.join(data_folder, mat_file[index][-1].item()) : mat_file[index][-2].item() - 1 for index in range(len(mat_file))}

        # The mean and std is from imagenet
        mean = [0,0,0]
        std = [1,1,1]
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((imsize, imsize)),
                                             transforms.Normalize(mean=mean, std=std)])

        self.ERROR_COUNT = 0

    def load_image(self, im_file):
        return Image.open(im_file)

    def __len__(self) -> int:
        return len(self.im_files)

    def __getitem__(self, index) -> ("torch.tensor", "torch.tensor"):
        if self.ERROR_COUNT > 20: raise Exception("DataLoader causes error 20 times continuously!")
        img = self.load_image(self.im_files[index])
        label = self.labels[self.im_files[index]]
        try:
            self.ERROR = 0
            return self.transform(img), torch.tensor(label)
        except:
            self.ERROR_COUNT += 1
            return self.__getitem__(index + 1)