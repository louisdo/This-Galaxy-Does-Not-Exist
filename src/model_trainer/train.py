import torch, torchvision
import logging, time, sys
sys.path.append("../..")
from loader import Galaxy10Dataset
from src.model.discriminator import Discriminator
from src.model.generator import Generator
from utils.utils import Utils
from tqdm import tqdm


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

        self.fixed_noise = torch.randn(64, self.CONFIG["latent_dim"], 1, 1, device = self.device)

        self.image_list = []

    def _update_discriminator(self, 
                              images: torch.tensor,
                              generated_images: torch.tensor, 
                              labels: torch.tensor, 
                              fake_labels: torch.tensor):
        # input images into discriminator, the discriminator is
        # expected to classify all those images as real
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

        return averaged_discriminator_loss


    def _update_generator(self, 
                          generated_images: torch.tensor, 
                          labels: torch.tensor):
        self.generator.zero_grad()
        discriminator_loss_for_fake_images = self.discriminator(generated_images).view(-1)

        # this means the generator should generate images that can fool the discriminator
        generator_loss = self.criterion(discriminator_loss_for_fake_images, labels)
        averaged_generator_loss = generator_loss.mean().item()
        generator_loss.backward()
        self.generator_optimizer.step()

        return averaged_generator_loss


    def _save_generator_output(self):
        with torch.no_grad():
            generated = self.generator(self.fixed_noise).detach().cpu()
        self.image_list.append(torchvision.utils.make_grid(generated, padding=2, normalize=True))


    def train_one_epoch(self, epoch: int):
        latent_dim = self.CONFIG["latent_dim"]
        eval_interval = self.CONFIG["eval_interval"]

        train_progress_bar = tqdm(self.dataloader, desc = f"Epoch {epoch}")

        for batch_index, images in enumerate(train_progress_bar):
            images = images.to(self.device)
            labels = torch.ones((images.size(0), )).to(self.device)
            fake_labels = torch.zeros((images.size(0), )).to(self.device)

            # randomly generate noise, this will be the input of the
            # generator
            noise = torch.randn(images.size(0), latent_dim, 1, 1).to(self.device)
            generated_images = self.generator(noise)

            # The first part is to update the discriminator
            averaged_discriminator_loss = self._update_discriminator(images = images,
                                                                     generated_images = generated_images,
                                                                     labels = labels,
                                                                     fake_labels = fake_labels)

            # The second part is to update the generator
            averaged_generator_loss = self._update_generator(generated_images = generated_images,
                                                             labels = labels)

            # update the training progress bar
            train_progress_bar.set_postfix({
                "Generator loss": averaged_generator_loss,
                "Discriminator_loss": averaged_discriminator_loss
            })


            if batch_index != 0 and batch_index % eval_interval:
                self._save_generator_output()


    def train(self):
        num_epochs = self.CONFIG["num_epochs"]

        training_start_time = time.time()
        logging.info(f"Starting training {num_epochs} epochs at {training_start_time}")

        for epoch in range(num_epochs):
            self.train_one_epoch(epoch)

        training_end_time = time.time()
        logging.info(f"The training procedure took {(training_end_time - training_start_time) / 60} minutes")




if __name__ == "__main__":
    CONFIG = Utils.get_config_yaml("../../config/config.yml")
    model_trainer = ModelTrainer(CONFIG = CONFIG)

    model_trainer.train()