"""This file contains the dataloader"""
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from astroNN.datasets import galaxy10
from torchvision import transforms


class Galaxy10Dataset(Dataset):
    def __init__(self, imsize: int = 64) -> None:
        # This will download the data at the first time being run
        images, labels = galaxy10.load_data()

        images = images.astype(np.float32)

        self.images = self._resize_array_of_images(images, (imsize, imsize, 3))
        self.labels = labels

        # The mean and std is from imagenet
        mean = [123.675, 116.28 , 103.53]
        std = [58.395, 57.12 , 57.375]
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])



    def _resize_array_of_images(self, images: np.array, shapes: list) -> np.array:
        height, width, channels = shapes
        resized_images = np.zeros((len(images), height, width, channels))

        for index in tqdm(range(len(images)), desc = "Resizing the images"):
            image = images[index]
            resized_image = cv2.resize(image, (height, width))
            resized_images[index] += resized_image

        return resized_images
        

    def __len__(self) -> int:
        return len(self.images)


    def __getitem__(self, index) -> "torch.tensor":
        return self.transform(self.images[index])