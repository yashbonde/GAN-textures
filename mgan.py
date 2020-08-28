
import os
import time
import random
import argparse
import numpy as np
from tqdm import trange
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils import tensorboard as tb

from maze import Maze

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
parser.add_argument("--model_folder", type = str, default = "models", help = "folder to save model to")
parser.add_argument("--save_every", type = int, default = 50, help = "interval to save the models")
parser.add_argument("--seed", type = int, default = 4, help = "seed value")
opt = parser.parse_args()
opt = SimpleNamespace(**vars(opt), img_size = opt.size + int(opt.size % 2 == 0))

cuda = True if torch.cuda.is_available() else False
os.makedirs(opt.model_folder, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(opt.seed)

class Mazze():
    def __init__(self, w, h):
        self.width = int(w/2)
        self.height = int(h/2)

    def __len__(self):
        # just any random number
        return 10000

    def __getitem__(self, *args, **kwargs):
        m = Maze().generate(width=self.width, height=self.height)
        m = m._to_str_matrix(_np = True)
        m = torch.from_numpy(m)
        return m

    def __iter__(self):
        m = Maze().generate(width=self.width, height=self.height)
        m = m._to_str_matrix(_np = True)
        m = torch.from_numpy(m)
        yield m


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, _pad = False):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        if _pad:
            new_size = img.size(2) + 1
            img_padded = torch.ones((img.size(0), 1, new_size, new_size))
            img_padded[:, :, 1:, 1:] = img
            return img_padded
        else:
            return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# print(f"Generator: {generator}")
# print(f"Discriminator: {discriminator}")

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
# print(f"Latent: {z.size()}")
# gen_imgs = generator(z)
# print(f"gen_imgs: {gen_imgs.size()}")
# gen_dis = discriminator(gen_imgs.detach())
# print(gen_dis.size())

# Configure data loader
m = Mazze(opt.size, opt.size)
# dataloader = DataLoader(m, batch_size = opt.batch_size)
# for i,m in enumerate(dataloader):
#     if i: break
#     print(m.shape)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

with tb.SummaryWriter(log_dir = opt.model_folder, flush_secs = 20) as sw:
    try:
        global_step = 0
        for epoch in range(opt.n_epochs):
            size = (len(m) // opt.batch_size) + int(len(m) % opt.batch_size != 1)
            pbar = trange(size)
            dataloader = DataLoader(m, batch_size = opt.batch_size)
            for i, imgs in zip(pbar, dataloader):
                b, x, y = imgs.shape
                imgs = imgs.view(b, 1, x, y)

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))[:,:,1:,1:]

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

                # Generate a batch of images
                gen_imgs = generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # run the discriminator on real and generated values
                gen_dis = discriminator(gen_imgs.detach())
                real_dis = discriminator(real_imgs)

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(real_dis, valid)
                fake_loss = adversarial_loss(gen_dis, fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                pbar.set_description(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                batches_done = epoch * len(dataloader) + i
                if batches_done % opt.sample_interval == 0:
                    print(f"saving images at: images/{batches_done}.png")
                    images_to_save = gen_imgs.data[:25]
                    images_to_save[images_to_save >= 0.5] = 1
                    images_to_save[images_to_save < 0.5] = 0
                    save_image(images_to_save, f"images/{batches_done}.png", nrow=5, normalize=True)

                if batches_done % opt.save_every == 0:
                    print(f"Saving model in folder: {opt.model_folder}")
                    torch.save(generator.state_dict(), f"{opt.model_folder}/generator.pt")
                    torch.save(discriminator.state_dict(), f"{opt.model_folder}/discriminator.pt")

                sw.add_scalar("Dis-Loss/Real", real_loss.item(), global_step = global_step, walltime = time.time())
                sw.add_scalar("Dis-Loss/Fake", fake_loss.item(), global_step = global_step, walltime = time.time())
                sw.add_scalar("Dis-Loss/Total", d_loss.item(), global_step = global_step, walltime = time.time())
                sw.add_scalar("Gen-Loss/Loss", g_loss.item(), global_step = global_step, walltime = time.time())

                gen_img_sharpened = gen_imgs[0].clone()
                gen_img_sharpened[gen_img_sharpened >= 0.5] = 1
                gen_img_sharpened[gen_img_sharpened < 0.5] = 0
                sw.add_image("Generated", gen_img_sharpened, global_step = global_step, walltime = time.time())
                sw.add_image("Real", real_imgs[0], global_step = global_step, walltime = time.time())

                global_step += 1
    except KeyboardInterrupt:
        pass

print(f"Saving model in folder: {opt.model_folder}")
torch.save(generator.state_dict(), f"{opt.model_folder}/generator.pt")
torch.save(discriminator.state_dict(), f"{opt.model_folder}/discriminator.pt")
