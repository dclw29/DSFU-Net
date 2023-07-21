# Copyright (c) 2022-2023 Lucas Rudden
#
# DSFU-Net is free software ;
# you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
# either version 2 of the License, or (at your option) any later version.
# DSFU-Net is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with DSFU-Net ;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
#
# Author : Lucas S. P. Rudden, lucas.s.p.rudden@gmail.com

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
curr_dir = os.getcwd()
sys.path.append(curr_dir)
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
    """
    For training!
    """
    def __init__(self, image, mode="train"):

        self.size = image.shape
        self.image = image.tobytes()

    def getimage(self):

        image = np.frombuffer(self.image, dtype=np.float32) #self.image 
        image = image.reshape(self.size)
        h, w = self.size
        image_A = image[:, : int(w / 3)]
        image_B = image[:, int(w / 3) : 2*int(w / 3)]
        image_C = image[:, 2*int(w / 3):]

        return image_A, image_B, image_C

def sample_images(batches_done, batch_dFF, batch_SRO, generator_dFF, generator_SRO, opt, save_loc):
    """Saves a generated sample - should be from validation set"""
    real_A_dFF = inverse_spread(Variable(batch_dFF["A"].type(Tensor)))
    real_B_dFF = inverse_spread(Variable(batch_dFF["B"].type(Tensor)))
    real_A_SRO = inverse_spread(Variable(batch_SRO["A"].type(Tensor)))
    real_B_SRO = inverse_spread(Variable(batch_SRO["B"].type(Tensor)))

    fake_B_dFF = generator_dFF(real_A_dFF)
    fake_B_SRO = generator_SRO(real_A_SRO)

    # need to double check ordering of datasamples
    img_sample_dFF = torch.cat((real_A_dFF.data, fake_B_dFF.data, real_B_dFF.data), -1) # create big sample to save
    img_sample_SRO = torch.cat((real_A_SRO.data, fake_B_SRO.data, real_B_SRO.data), -1)

    save_image(img_sample_dFF, save_loc + "/../images/%s/epoch%s_dFF.png" % (opt.experiment_name, str(batches_done)), nrow=5, normalize=True)
    save_image(img_sample_SRO, save_loc + "/../images/%s/epoch%s_SRO.png" % (opt.experiment_name, str(batches_done)), nrow=5, normalize=True)

def train_GD(optimizer_G, optimizer_D, batch, generator, discriminator, patch, criterion_GAN, criterion_pixelwise, opt):
    """
    Train the relevent generator and discriminator 
    """
    # Model inputs
    real_A = inverse_spread(Variable(batch["A"].type(Tensor)))
    real_B = inverse_spread(Variable(batch["B"].type(Tensor))) # this corresponds to the scattering data   

    real_C = inverse_spread(Variable(batch["C"].type(Tensor))) # non-artefact scattering data

    # Adversarial ground truths
    valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
    fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)    

    # ------------------
    #  Train Generators
    # ------------------

    optimizer_G.zero_grad()

    # GAN loss
    fake_B = generator(real_A)
    pred_fake = discriminator(fake_B, real_C) # real_A
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
    pred_real = discriminator(real_B, real_C) # try inputting non-artefact data into discrim
    loss_real = criterion_GAN(pred_real, valid)

    # Fake loss
    pred_fake = discriminator(fake_B.detach(), real_C)
    loss_fake = criterion_GAN(pred_fake, fake)

    # Total loss
    loss_D = 0.5 * (loss_real + loss_fake)

    loss_D.backward()
    optimizer_D.step()

    return optimizer_G, optimizer_D, generator, discriminator, [loss_D, loss_G, loss_pixel, loss_GAN]

def train_both_scat(optimizer_G_dFF, batch_dFF, generator_dFF, optimizer_G_SRO, batch_SRO, generator_SRO, Scat_Loss, opt):
    """
    Train based on generating a total scattering image (training both SRO and dFF at the same time), then calculate MSE vs real image to train generators
    """

    # Model inputs
    real_A_dFF = inverse_spread(Variable(batch_dFF["A"].type(Tensor))) # this corresponds to the scattering data   
    real_B_dFF = inverse_spread(Variable(batch_dFF["B"].type(Tensor))) # This is the target (the dFF or SRO data)
    real_A_SRO = inverse_spread(Variable(batch_SRO["A"].type(Tensor))) # real A from both dFF and SRO should be the same
    real_B_SRO = inverse_spread(Variable(batch_SRO["B"].type(Tensor))) 

    identity_test = real_A_dFF == real_A_SRO # these should be identical inputs

    assert ~torch.any(~identity_test), "Scattering data output from dFF and SRO not identical! Please check input"

    # ------------------
    #  Train Generators from smooth L1 loss and the known ground truth output
    # ------------------

    optimizer_G_dFF.zero_grad()
    optimizer_G_SRO.zero_grad()

    # GAN loss
    fake_B_dFF = generator_dFF(real_A_dFF)
    fake_B_SRO = generator_SRO(real_A_SRO) # create both images based on scattering data
    # generated output is between -1 and 1, we need to avoid making minima into maxima
    fake_A_Scat = (fake_B_dFF+2) * (fake_B_SRO+2) - 2

    # owing to the Cauchy–Schwarz inequality, These two outputs multiplied does not give the normalised actual scattering we're feeding into the network
    # Therefore, we need to compare the loss with the real multiple of the dFF and SRO normalised images
    real_Scat = (real_B_dFF+2) * (real_B_SRO+2) - 2
    loss = Scat_Loss(fake_A_Scat, real_Scat)
    loss.backward()

    optimizer_G_dFF.step()
    optimizer_G_SRO.step()

    return optimizer_G_dFF, generator_dFF, optimizer_G_SRO, generator_SRO, loss

def inverse_spread(data):
    """
    Lots of values are grouped up around -1 (or < -0.95), spread them out
    """

    min_shift_val=0.
    max_shift_val=2**0.5
    shifted_data = (data + 1)**0.5
    min_shift_val = shifted_data.min() # keep record to unnormalise later, max will still be 1
    renorm_data_tmp = shifted_data - min_shift_val
    max_shift_val = renorm_data_tmp.max()
    renorm_data = (renorm_data_tmp / max_shift_val) * 2 - 1
    return renorm_data

def reverse_spread(data):
    """
    Convert the data back into what we would expect
    """

    min_shift_val=0.
    max_shift_val=2**0.5

    # first unnorm
    unnorm_data_tmp = ((data + 1) / 2) * max_shift_val  # remember 1 would have been max before, so minus min from this
    unnorm_data = unnorm_data_tmp + min_shift_val
    # we can't account for the random added noise, so ignore that for now
    unshifted_data = unnorm_data**2 - 1
    return unshifted_data

# training class
class Train:
    def __init__(self, opt):

        self.opt = opt
        self.epoch = opt.epoch
        self.n_epochs = opt.n_epochs
        self.dataset_name = opt.dataset_name
        self.experiment_name = opt.experiment_name
        self.batch_size = opt.batch_size
        self.lr = opt.lr
        self.b1 = opt.b1
        self.b2 = opt.b2
        self.decay_epoch = opt.decay_epoch
        self.n_cpu = opt.n_cpu
        self.img_height = opt.img_height
        self.img_width = opt.img_width
        self.channels = opt.channels
        self.rank = opt.rank
        self.world_size = opt.world_size
        self.lambda_pixel = opt.lambda_pixel
        self.sample_interval = opt.sample_interval
        self.checkpoint_interval = opt.checkpoint_interval
        self.log_interval = opt.log_interval
        self.gpu = opt.gpu
        self.run_continue = opt.run_continue

        cuda = torch.device('cuda:{}'.format(self.gpu))

        # global variable - tensor type
        global Tensor
        if torch.cuda.is_available():
            Tensor = torch.cuda.FloatTensor
        else:
            Tensor = torch.FloatTensor
        device = torch.device('cuda:{}'.format(self.gpu))
        torch.cuda.set_device('cuda:{}'.format(self.gpu))

        # locations of training and test data
        file_loc = "%s/../"%curr_dir
        training_loc = self.dataset_name
        save_loc = file_loc + "/models/"
        self.save_loc = save_loc
        
        # from current directory, create models and images of network
        os.makedirs(file_loc + "/RUN/images/%s" % opt.experiment_name, exist_ok=True)
        os.makedirs(save_loc + "/%s" % opt.experiment_name, exist_ok=True)

        # Loss functions
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_pixelwise = torch.nn.L1Loss()
        # L2 loss more prone to outliers, use L2 for faster optimisation, but L1 when we have possible outlier
        self.Scat_Loss = torch.nn.SmoothL1Loss() # https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html

        self.patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

        # Initialize generators and discriminators
        self.generator_dFF = GeneratorUNet(in_channels=self.channels, out_channels=self.channels)
        self.discriminator_dFF = Discriminator(in_channels=self.channels)
        self.generator_SRO = GeneratorUNet(in_channels=self.channels, out_channels=self.channels)
        self.discriminator_SRO = Discriminator(in_channels=self.channels)
        
        if cuda:
            self.generator_dFF = self.generator_dFF.cuda()
            self.discriminator_dFF = self.discriminator_dFF.cuda()
            self.generator_SRO = self.generator_SRO.cuda()
            self.discriminator_SRO = self.discriminator_SRO.cuda()
            self.criterion_GAN.cuda()
            self.criterion_pixelwise.cuda()
            self.Scat_Loss.cuda()

        # continue from a pre-existing network
        if opt.run_continue:
            # Load pretrained models
            self.generator_dFF.load_state_dict(torch.load(save_loc + "/%s/generator_dFF_%d.pth" % (self.experiment_name, self.epoch)))
            self.discriminator_dFF.load_state_dict(torch.load(save_loc + "/%s/discriminator_dFF_%d.pth" % (self.experiment_name, self.epoch)))
            self.generator_SRO.load_state_dict(torch.load(save_loc + "/%s/generator_SRO_%d.pth" % (self.experiment_name, self.epoch)))
            self.discriminator_SRO.load_state_dict(torch.load(save_loc + "/%s/discriminator_SRO_%d.pth" % (self.experiment_name, self.epoch)))
        else: # WON'T WORK WITH SELF ATTN BECAUSE OF SPEC NORM
            # Initialize weights
            self.generator_dFF.apply(weights_init_normal)
            self.discriminator_dFF.apply(weights_init_normal)
            self.generator_SRO.apply(weights_init_normal)
            self.discriminator_SRO.apply(weights_init_normal)

        # Optimizers
        self.optimizer_G_dFF = torch.optim.Adam(self.generator_dFF.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D_dFF = torch.optim.Adam(self.discriminator_dFF.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_G_SRO = torch.optim.Adam(self.generator_SRO.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D_SRO = torch.optim.Adam(self.discriminator_SRO.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        
        transforms_ = [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
        
        self.dataset_dFF = LMDBDataset(root=training_loc + '/train_lmdb_dFF', 
                              name='scattering', train=True, transform=transforms_, is_encoded=False)
        self.data_loader_dFF = torch.utils.data.DataLoader(self.dataset_dFF,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=self.n_cpu,
                                                      pin_memory=True)
        

        self.dataset_SRO = LMDBDataset(root=training_loc + '/train_lmdb_SRO', 
                              name='scattering', train=True, transform=transforms_, is_encoded=False)
        self.data_loader_SRO = torch.utils.data.DataLoader(self.dataset_SRO,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=self.n_cpu,
                                                      pin_memory=True)

    def train(self):
        """
        Train the network based on self parameters
        """

        prev_time = time.time()
        if self.run_continue:
            self.epoch += 1

        for epoch in range(self.epoch, self.n_epochs):
            for i, batch in enumerate(zip(self.data_loader_dFF, self.data_loader_SRO)):

                batch_dFF, batch_SRO = batch
        
                self.optimizer_G_dFF, self.optimizer_D_dFF, self.generator_dFF, self.discriminator_dFF, loss_dFF = train_GD(self.optimizer_G_dFF, self.optimizer_D_dFF, batch_dFF, self.generator_dFF, self.discriminator_dFF, self.patch, self.criterion_GAN, self.criterion_pixelwise, self.opt)
                self.optimizer_G_SRO, self.optimizer_D_SRO, self.generator_SRO, self.discriminator_SRO, loss_SRO = train_GD(self.optimizer_G_SRO, self.optimizer_D_SRO, batch_SRO, self.generator_SRO, self.discriminator_SRO, self.patch, self.criterion_GAN, self.criterion_pixelwise, self.opt)
        
                self.optimizer_G_dFF, self.generator_dFF, self.optimizer_G_SRO, self.generator_SRO, loss_scat = train_both_scat(self.optimizer_G_dFF, batch_dFF, self.generator_dFF, self.optimizer_G_SRO, batch_SRO, self.generator_SRO, self.Scat_Loss, self.opt)

                batches_done = epoch * len(self.data_loader_dFF) + i
                if batches_done % self.log_interval == 0:

                    #################################################################################
                    # --------------
                    #  Log Progress
                    # --------------
        
                    # Determine approximate time left
                    batches_left = opt.n_epochs * len(self.data_loader_dFF) - batches_done
                    time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                    prev_time = time.time()
        
                    # Print log
                    sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [dFF D loss: %f] [dFF G loss: %f, pixel: %f, adv: %f] [SRO D loss: %f] [SRO G loss: %f, pixel: %f, adv: %f] [Scat loss: %f] ETA: %s\n"
                        % (
                            epoch,
                            opt.n_epochs,
                            i,
                            len(self.data_loader_dFF),
                            loss_dFF[0].item(),
                            loss_dFF[1].item(),
                            loss_dFF[2].item(),
                            loss_dFF[3].item(),
                            loss_SRO[0].item(),
                            loss_SRO[1].item(),
                            loss_SRO[2].item(),
                            loss_SRO[3].item(),
                            loss_scat.item(),
                            time_left,
                        )
                    )
                    
                    sys.stdout.flush() 

            # if at end of epoch, save image
            sample_images(epoch, batch_dFF, batch_SRO, self.generator_dFF, self.generator_SRO, opt, self.save_loc)
        
            if self.checkpoint_interval != -1 and epoch % self.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(self.generator_dFF.state_dict(), self.save_loc + "/%s/generator_dFF_%d.pth" % (self.experiment_name, epoch))
                torch.save(self.discriminator_dFF.state_dict(), self.save_loc + "/%s/discriminator_dFF_%d.pth" % (self.experiment_name, epoch))
                torch.save(self.generator_SRO.state_dict(), self.save_loc + "/%s/generator_SRO_%d.pth" % (self.experiment_name, epoch))
                torch.save(self.discriminator_SRO.state_dict(), self.save_loc + "/%s/discriminator_SRO_%d.pth" % (self.experiment_name, epoch))                

################################################################################################################
####### MAIN RUN CODE #######

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200+1, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="/path/to/training/data/DSFU-Net/dataset/training/", help="name of the dataset")
    parser.add_argument("--experiment_name", type=str, default="first_test", help="name of expt")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument(
        "--sample_interval", type=int, default=1000, help="interval between sampling of images from generators"
    )
    parser.add_argument("--log_interval", type=int, default=100, help="How often to post log messages")
    parser.add_argument("--checkpoint_interval", type=int, default=4, help="interval between model checkpoints")
    parser.add_argument("--rank", type=int, default=0, help="GPU rank of process")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of GPUs being trained on")
    parser.add_argument("--lambda_pixel", type=int, default=100, help="Lambda pixel smoother weight")

    parser.add_argument('--run_continue', default=False, action='store_true', help='Continuing from an old run?')

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    opt = parser.parse_args()

    T = Train(opt)
    T.train()
