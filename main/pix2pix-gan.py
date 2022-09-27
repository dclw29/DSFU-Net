"""
Adapted from: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/pix2pix.py
"""

import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
#file_path = os.path.realpath(__file__)
#file_loc = file_path[:-14]
#sys.path.append(file_loc)
sys.path.append("/home/dclw/ML-DiffuseReader/main")
from prepare_lmdb_input import LMDBDataset
from argparse import Namespace
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch

class LMDB_Image:
    def __init__(self, image, mode="train"):

        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()

        #TODO need test information also

    def getimage(self):

        image = np.frombuffer(self.image, dtype=np.uint8)
        image = image.reshape(self.size + (self.channels,))
        h, w = self.size 
        image_A = image[:, : int(w / 2), :]
        image_B = image[:, int(w / 2) :, :]
        #image_A = image.crop((0, 0, w / 2, h)) # note w/2 here, in other words, target image needs to be on the right
        #image_B = image.crop((w / 2, 0, w, h))  
        image_A = Image.fromarray(image_A).convert("RGB") #, mode="RGB")
        image_B = Image.fromarray(image_B).convert("RGB") #, mode="RGB")

        return image_A, image_B  

# PARAMS without needing the flag arguments
args = {
        'epoch' : 0,
        'n_epochs' : 5,
        'dataset_name' : "/home/dclw/ML-DiffuseReader/dataset/training",
        'batch_size' : 32,
        'lr' : 0.0002,
        'b1' : 0.5,
        'b2' : 0.999,
        'decay_epoch' : 100,
        'n_cpu' : 1,
        'img_height' : 256,
        'img_width' : 256,
        'channels' : 3,
        'rank' : 0,
        'world_size' : 1, # world size as in number of ranks overall (number of GPUs)
        'lambda_pixel' : 100 # Loss weight of L1 pixel-wise loss between translated image and real image
}

opt = Namespace(**args)

# locations of training and test data
file_loc = "/home/dclw/ML-DiffuseReader"
training_loc = opt.dataset_name
test_loc = file_loc + "/dataset/test"
save_loc = file_loc + "/RUN/saved_models"

# from current directory, create models and images of network
os.makedirs(file_loc + "/RUN/images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs(save_loc + "/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generators and discriminators
generator_dFF = GeneratorUNet()
discriminator_dFF = Discriminator()
generator_SRO = GeneratorUNet()
discriminator_SRO = Discriminator()

if cuda:
    generator_dFF = generator_dFF.cuda()
    discriminator_dFF = discriminator_dFF.cuda()
    generator_SRO = generator_SRO.cuda()
    discriminator_SRO = discriminator_SRO.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

# continue from a pre-existing network
if opt.epoch != 0:
    # Load pretrained models
    generator_dFF.load_state_dict(torch.load(save_loc + "/%s/generator_dFF_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator_dFF.load_state_dict(torch.load(save_loc + "/%s/discriminator_dFF_%d.pth" % (opt.dataset_name, opt.epoch)))
    generator_SRO.load_state_dict(torch.load(save_loc + "/%s/generator_SRO_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator_SRO.load_state_dict(torch.load(save_loc + "/%s/discriminator_SRO_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator_dFF.apply(weights_init_normal)
    discriminator_dFF.apply(weights_init_normal)
    generator_SRO.apply(weights_init_normal)
    discriminator_SRO.apply(weights_init_normal)

# Optimizers
optimizer_G_dFF = torch.optim.Adam(generator_dFF.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_dFF = torch.optim.Adam(discriminator_dFF.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G_SRO = torch.optim.Adam(generator_SRO.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_SRO = torch.optim.Adam(discriminator_SRO.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

#TODO: configure dataloader
# Configure dataloaders
#transforms_ = [
#    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#]
# try without resizing first
transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataset_dFF = LMDBDataset(root=training_loc + '/train_lmdb_dFF', 
                      name='scattering', train=True, transform=transforms_, is_encoded=False)
#TODO setup distributed training
train_sampler_dFF = torch.utils.data.distributed.DistributedSampler(dataset_dFF,
                                                                num_replicas=opt.world_size,
                                                                rank=opt.rank)
data_loader_dFF = torch.utils.data.DataLoader(dataset_dFF,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=opt.n_cpu,
                                              pin_memory=True,
                                              sampler=train_sampler_dFF,
                                              drop_last = True)

dataset_SRO = LMDBDataset(root=training_loc + '/train_lmdb_SRO', 
                      name='scattering', train=True, transform=transforms_, is_encoded=False)
#TODO setup distributed training
train_sampler_SRO = torch.utils.data.distributed.DistributedSampler(dataset_SRO,
                                                                num_replicas=opt.world_size,
                                                                rank=opt.rank)
data_loader_SRO = torch.utils.data.DataLoader(dataset_SRO,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=opt.n_cpu,
                                              pin_memory=True,
                                              sampler=train_sampler_SRO,
                                              drop_last = True)

# validation dataset
#val_dataloader = DataLoader(
#    ImageDataset(test_loc, transforms_=transforms_, mode="val"),
#    batch_size=10,
#    shuffle=True,
#    num_workers=1,
#)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["B"].type(Tensor))
    real_B = Variable(imgs["A"].type(Tensor))
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)

def train_GD(optimizer_G, optimizer_D, batch, generator, discriminator, opt):
    """
    Train the relevent generator and discriminator 
    """
    # Model inputs
    real_A = Variable(batch["A"].type(Tensor))
    real_B = Variable(batch["B"].type(Tensor)) # this corresponds to the scattering data   

    # Adversarial ground truths
    valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
    fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)    

    # ------------------
    #  Train Generators
    # ------------------

    optimizer_G.zero_grad()

    # GAN loss
    fake_B = generator(real_A)
    pred_fake = discriminator(fake_B, real_A)
    loss_GAN = criterion_GAN(pred_fake, valid)
    # Pixel-wise loss
    loss_pixel = criterion_pixelwise(fake_B, real_B)

    # Total loss
    loss_G = loss_GAN + opt.lambda_pixel * loss_pixel

    loss_G.backward()

    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Real loss
    pred_real = discriminator(real_B, real_A)
    loss_real = criterion_GAN(pred_real, valid)

    # Fake loss
    pred_fake = discriminator(fake_B.detach(), real_A)
    loss_fake = criterion_GAN(pred_fake, fake)

    # Total loss
    loss_D = 0.5 * (loss_real + loss_fake)

    loss_D.backward()
    optimizer_D.step()

    return optimizer_G, optimizer_D, generator, discriminator

# ----------
#  Training
#  We need to train two different images on two different outputs, then multiply those to get the original input
#  Train 1 at a time first, then after training retrain with multplier loss
#  You will need to do ablation tests on this if it works
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(zip(data_loader_dFF, data_loader_SRO)):
        batch_dFF, batch_SRO = batch

        optimizer_G_dFF, optimizer_D_dFF, generator_dFF, discriminator_dFF = train_GD(optimizer_G_dFF, optimizer_D_dFF, batch_dFF, generator_dFF, discriminator_dFF, opt)
        optimizer_G_SRO, optimizer_D_SRO, generator_SRO, discriminator_SRO = train_GD(optimizer_G_SRO, optimizer_D_SRO, batch_SRO, generator_SRO, discriminator_SRO, opt)

        #TODO add additional loss here combining dFF and SRO to scattering output

        #TODO add good logging here






        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="diffuse", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument(
        "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    opt = parser.parse_args()