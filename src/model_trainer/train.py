import torch, torchvision
import logging, time, sys, os
sys.path.append("../..")
import numpy as np
import pandas as pd
from loader import Galaxy10Dataset
from src.model.discriminator import Discriminator
from src.model.generator import Generator
from utils.utils import Utils
from tqdm import tqdm
from PIL import Image


class ModelTrainer:
    def __init__(self, 
                 CONFIG: dict):
        self.CONFIG = CONFIG["train"]
        self.device = torch.device(self.CONFIG["device"])

        self.discriminator = Discriminator(number_channel = 3,
                                           image_size = self.CONFIG["image_size"],
                                           ngpu = False)

        self.discriminator = self.discriminator.to(self.device)

        self.generator = Generator(latent_dim = self.CONFIG["latent_dim"],
                                   number_channel = 3,
                                   image_size = self.CONFIG["image_size"],
                                   ngpu = False)

        self.generator = self.generator.to(self.device)

        dataset = Galaxy10Dataset(imsize = self.CONFIG["image_size"])

        self.dataloader = torch.utils.data.DataLoader(dataset, 
                                                      batch_size = self.CONFIG["batch_size"],
                                                      shuffle = True, 
                                                      num_workers = self.CONFIG["workers"])

        self.criterion = torch.nn.BCELoss()

        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), 
                                                        lr=self.CONFIG["learning_rate"], 
                                                        betas=(0.5, 0.999))

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), 
                                                    lr=self.CONFIG["learning_rate"], 
                                                    betas=(0.5, 0.999))

        self.fixed_noise = torch.randn(1, self.CONFIG["latent_dim"], 1, 1, device = self.device)

        self.image_list = []


    def _save_generator_output(self):
        with torch.no_grad():
            generated = self.generator(self.fixed_noise).detach().cpu()
        self.image_list.append(torchvision.utils.make_grid(generated, padding=2, normalize=True))


    def train_one_epoch(self, epoch: int):
        latent_dim = self.CONFIG["latent_dim"]
        eval_interval = self.CONFIG["eval_interval"]

        train_progress_bar = tqdm(self.dataloader, desc = f"Epoch {epoch}")

        for batch_index, images in enumerate(train_progress_bar):
            images = images.float().to(self.device)
            labels = torch.ones((images.size(0), )).float().to(self.device)
            fake_labels = torch.zeros((images.size(0), )).float().to(self.device)

            # randomly generate noise, this will be the input of the
            # generator
            noise = torch.randn(images.size(0), latent_dim, 1, 1).float().to(self.device)
            generated_images = self.generator(noise)

            assert images.shape == generated_images.shape, "generated images and images must be of the same shape"
            assert labels.shape == fake_labels.shape, "labels and fake labels must be of the same shape"

            # The first part is to update the discriminator
            self.discriminator.zero_grad()
            discriminator_output = self.discriminator(images).view(-1)

            discriminator_loss = self.criterion(discriminator_output, labels)
            discriminator_loss.backward()

            discriminator_output_for_fake_images = self.discriminator(generated_images.detach()).view(-1)

            discriminator_loss_for_fake_images = self.criterion(discriminator_output_for_fake_images, fake_labels)
            discriminator_loss_for_fake_images.backward()

            total_loss = discriminator_loss + discriminator_loss_for_fake_images
            averaged_discriminator_loss = total_loss.mean().item()
            self.discriminator_optimizer.step()

            # The second part is to update the generator
            self.generator.zero_grad()
            discriminator_loss_for_fake_images = self.discriminator(generated_images).view(-1)

            # this means the generator should generate images that can fool the discriminator
            generator_loss = self.criterion(discriminator_loss_for_fake_images, labels)
            averaged_generator_loss = generator_loss.mean().item()
            generator_loss.backward()
            self.generator_optimizer.step()

            # update the training progress bar
            train_progress_bar.set_postfix({
                "Generator loss": averaged_generator_loss,
                "Discriminator loss": averaged_discriminator_loss
            })


            if batch_index != 0 and batch_index % eval_interval == 0:
                self._save_generator_output()


    def train(self):
        num_epochs = self.CONFIG["num_epochs"]

        training_start_time = time.time()
        logging.info(f"Starting training {num_epochs} epochs at {training_start_time}")

        for epoch in range(num_epochs):
            self.train_one_epoch(epoch)

        training_end_time = time.time()
        logging.info(f"The training procedure took {(training_end_time - training_start_time) / 60} minutes")

    def save_result_for_eval(self, num_infer, where_to):
        filenames = []
        for index in tqdm(range(num_infer), desc = "Infering for evaluation"):
            with torch.no_grad():
                noise = torch.randn(1, self.CONFIG["latent_dim"], 1, 1, device = self.device)
                generated = self.generator(noise).detach().cpu()
                generated = torchvision.utils.make_grid(generated, padding=2, normalize=True) * 255
                generated = generated.permute(1, 2, 0).cpu().detach().numpy()
                
            im = Image.fromarray(generated.astype(np.uint8))
            fname = f"fake_{index}.jpg"
            im.save(os.path.join(where_to, fname))
            filenames.append(fname)

        df = pd.DataFrame()
        df["path"] = filenames
        df.to_csv(os.path.join(where_to, "fake_data.csv"), index = False)



if __name__ == "__main__":
    CONFIG = Utils.get_config_yaml("../../config/config.yml")
    model_trainer = ModelTrainer(CONFIG = CONFIG)

    model_trainer.train()