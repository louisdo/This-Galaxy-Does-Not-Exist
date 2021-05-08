import torch, torchvision
import logging, time, sys, os
sys.path.append("../..")
import numpy as np
import pandas as pd
from loader import Galaxy10Dataset, StanfordCarDataset
from src.model.discriminator import Discriminator
from src.model.generator import Generator
from utils.utils import Utils
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser


class ModelTrainer:
    def __init__(self, 
                 CONFIG: dict):
        self.CONFIG = CONFIG["train"]
        self.device = torch.device(self.CONFIG["device"])

        NUM_CLASSES = {
            "car": 196,
            "galaxy": 10
        }
        self.discriminator = Discriminator(number_channel = 3,
                                           image_size = self.CONFIG["image_size"],
                                           num_classes = NUM_CLASSES[self.CONFIG["input"]],
                                           ngpu = False)
        if os.path.exists(self.CONFIG["resume_dis_ckpt"]): 
            self.discriminator.load_state_dict(torch.load(self.CONFIG["resume_dis_ckpt"]))
        self.discriminator = self.discriminator.to(self.device)

        self.generator = Generator(latent_dim = self.CONFIG["latent_dim"],
                                   number_channel = 3,
                                   image_size = self.CONFIG["image_size"],
                                   ngpu = False)
        if os.path.exists(self.CONFIG["resume_gen_ckpt"]): 
            self.generator.load_state_dict(torch.load(self.CONFIG["resume_gen_ckpt"]))
        self.generator = self.generator.to(self.device)

        if self.CONFIG["input"] == "galaxy": dataset = Galaxy10Dataset(imsize = self.CONFIG["image_size"])
        elif self.CONFIG["input"] == "car": dataset = StanfordCarDataset(data_folder = self.CONFIG["data_folder"],
                                                                        labels_file = self.CONFIG["labels_file"],
                                                                        imsize = self.CONFIG["image_size"])

        self.dataloader = torch.utils.data.DataLoader(dataset, 
                                                      batch_size = self.CONFIG["batch_size"],
                                                      shuffle = True, 
                                                      num_workers = self.CONFIG["workers"])

        self.criterion = torch.nn.BCELoss()
        self.classification_criterion = torch.nn.CrossEntropyLoss()

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

        train_progress_bar = tqdm(self.dataloader, desc = f"Epoch {epoch + 1}")

        self.generator.train()
        self.discriminator.train()

        for batch_index, (images, cls_labels) in enumerate(train_progress_bar):
            images = images.float().to(self.device)
            cls_labels = cls_labels.float().to(self.device)
            labels = torch.ones((images.size(0), )).float().to(self.device) * 0.9
            fake_labels = torch.ones((images.size(0), )).float().to(self.device) * 0.1

            # randomly generate noise, this will be the input of the
            # generator
            noise = torch.randn(images.size(0), latent_dim, 1, 1).float().to(self.device)
            generated_images = self.generator(noise)

            assert images.shape == generated_images.shape, "generated images and images must be of the same shape"
            assert labels.shape == fake_labels.shape, "labels and fake labels must be of the same shape"

            # The first part is to update the discriminator
            self.discriminator.zero_grad()
            #discriminator_output, pred_labels = self.discriminator(images, True)
            discriminator_output = self.discriminator(images, False)
            discriminator_output = discriminator_output.view(-1)
            #pred_labels = pred_labels.reshape(pred_labels.shape[:2])

            discriminator_loss = self.criterion(discriminator_output, labels) #+ self.classification_criterion(pred_labels, cls_labels.long())
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
            fname = os.path.join(where_to, f"fake_{index}.jpg")
            im.save(fname)
            filenames.append(fname)

        df = pd.DataFrame()
        df["path"] = filenames
        df.to_csv(os.path.join(where_to, "fake_data.csv"), index = False)

    def save_checkpoint(self):
        torch.save(self.generator.state_dict(), os.path.join(self.CONFIG["ckpt_folder"], "checkpoint.pth.tar"))
        return True


if __name__ == "__main__":
    parser = ArgumentParser(description = "Model trainer")
    parser.add_argument("--ckpt_folder", help = "path to save checkpoint", required = True)
    parser.add_argument("--eval_folder", help = "folder to where the eval data will be saved", required = True)
    args = parser.parse_args()

    CONFIG = Utils.get_config_yaml("../../config/config.yml")
    model_trainer = ModelTrainer(CONFIG = CONFIG)

    model_trainer.train()
    model_trainer.save_result_for_eval(25000, args.eval_folder)

    checkpoint_path = os.path.join(args.ckpt_folder, "checkpoint.pth.tar")
    torch.save(model_trainer.generator.state_dict(), checkpoint_path)
    print("Checkpoint saved")