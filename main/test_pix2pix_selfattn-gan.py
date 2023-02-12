"""
Test our PIX2PIX gan network based on validation dataset
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
sys.path.append("/data/lrudden/ML-DiffuseReader/main")
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
        self.size = image.shape
        self.image = image.tobytes()

        #TODO need test information also

    def getimage(self):

        image = np.frombuffer(self.image, dtype=np.float32)
        image = image.reshape(self.size)
        h, w = self.size
        image_A = image[:, : int(w / 2)]
        image_B = image[:, int(w / 2) :]

        return image_A, image_B

def sample_images(batches_done, batch_dFF, batch_SRO, generator_dFF, generator_SRO, opt, save_loc):
    """Saves a generated sample - should be from validation set"""
    real_A_dFF = Variable(batch_dFF["A"].type(Tensor))
    real_B_dFF = Variable(batch_dFF["B"].type(Tensor))
    real_A_SRO = Variable(batch_SRO["A"].type(Tensor))
    real_B_SRO = Variable(batch_SRO["B"].type(Tensor))

    fake_B_dFF, _, _ = generator_dFF(real_A_dFF)
    fake_B_SRO, _, _ = generator_SRO(real_A_SRO)

    real_scat = (real_B_dFF+2) * (real_B_SRO+2) - 2
    fake_scat = (fake_B_dFF+2) * (fake_B_SRO+2) -2

    #torch.save(real_B_dFF, "real_dFF.pt"); torch.save(real_B_SRO, "real_SRO.pt")
    #torch.save(fake_B_dFF, "fake_dFF.pt"); torch.save(fake_B_SRO, "fake_SRO.pt")
    #torch.save(real_A_dFF.data, "real_scat.pt")

    # check whether real scattering output looks reasonable for our network
    save_image(torch.cat((real_A_dFF.data, fake_scat, real_scat), -1), save_loc + "/%s/scat_mockup%s.png" % (opt.experiment_name, str(batches_done)), nrow=5, normalize=True)

    # need to double check ordering of datasamples
    img_sample_dFF = torch.cat((real_A_dFF.data, fake_B_dFF.data, real_B_dFF.data), -1) # create big sample to save
    img_sample_SRO = torch.cat((real_A_SRO.data, fake_B_SRO.data, real_B_SRO.data), -1)

    save_image(img_sample_dFF, save_loc + "/%s/slice%s_dFF.png" % (opt.experiment_name, str(batches_done)), nrow=5, normalize=True)
    save_image(img_sample_SRO, save_loc + "/%s/slice%s_SRO.png" % (opt.experiment_name, str(batches_done)), nrow=5, normalize=True)

    # save raw data also
    np.save(save_loc + "/%s/slice%s_dFF.npy" % (opt.experiment_name, str(batches_done)), torch.cat((real_A_dFF.data, fake_B_dFF.data, real_B_dFF.data), -1).detach().cpu().numpy())
    np.save(save_loc + "/%s/slice%s_SRO.npy" % (opt.experiment_name, str(batches_done)), torch.cat((real_A_SRO.data, fake_B_SRO.data, real_B_SRO.data), -1).detach().cpu().numpy())

# training class
class Test:
    def __init__(self, opt):

        self.opt = opt
        self.epoch = opt.epoch # which epoch to load in
        self.dataset_name = opt.dataset_name
        self.experiment_name = opt.experiment_name
        self.batch_size = opt.batch_size
        self.img_height = opt.img_height
        self.img_width = opt.img_width
        self.channels = opt.channels
        self.rank = opt.rank
        self.world_size = opt.world_size
        self.gpu = opt.gpu

        cuda = torch.device('cuda:{}'.format(self.gpu))
        #cuda = torch.device('cuda:1')  # cuda device
        # global variable - tensor type
        global Tensor
        if torch.cuda.is_available():
            Tensor = torch.cuda.FloatTensor
        else:
            Tensor = torch.FloatTensor
        device = torch.device('cuda:{}'.format(self.gpu))
        torch.cuda.set_device('cuda:{}'.format(self.gpu))

        # locations of validation set
        file_loc = "/data/lrudden/ML-DiffuseReader"
        validation_loc = self.dataset_name
        image_loc = file_loc + "/RUN/validation"
        self.image_loc = image_loc
        os.makedirs(image_loc + "/%s" % opt.experiment_name, exist_ok=True)
        save_loc = file_loc + "/RUN/saved_models"
        self.save_loc = save_loc
        
        # Initialize generators
        self.generator_dFF = GeneratorUNet_Attn(in_channels=self.channels, out_channels=self.channels)
        self.generator_SRO = GeneratorUNet_Attn(in_channels=self.channels, out_channels=self.channels)
        
        if cuda:
            self.generator_dFF = self.generator_dFF.cuda()
            self.generator_SRO = self.generator_SRO.cuda()

        # Load a pre-existing network
        self.generator_dFF.load_state_dict(torch.load(save_loc + "/%s/generator_dFF_%d.pth" % (self.experiment_name, self.epoch)))
        self.generator_SRO.load_state_dict(torch.load(save_loc + "/%s/generator_SRO_%d.pth" % (self.experiment_name, self.epoch)))

        transforms_ = [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
        
        self.dataset_dFF = LMDBDataset(root=validation_loc + '/val_lmdb_dFF', 
                              name='scattering', train=False, transform=transforms_, is_encoded=False)

        self.data_loader_dFF = torch.utils.data.DataLoader(self.dataset_dFF,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=0,
                                                      pin_memory=True,
                                                      drop_last=True)

        self.dataset_SRO = LMDBDataset(root=validation_loc + '/val_lmdb_SRO', 
                              name='scattering', train=False, transform=transforms_, is_encoded=False)

        self.data_loader_SRO = torch.utils.data.DataLoader(self.dataset_SRO,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=0,
                                                      pin_memory=True,
                                                      drop_last = True)

    def test(self):
        """
        Test across batch images. Take a random slice out from them, but calculate loss across all...? At different points during training? 
        """

        num_possible_iters = len(self.data_loader_dFF)
        STOP = np.random.randint(num_possible_iters)

        for i, batch in enumerate(zip(self.data_loader_dFF, self.data_loader_SRO)):

            if i != STOP:
                pass
            else:

                batch_dFF, batch_SRO = batch
        
                # Need to calculate all losses, and likely FID etc. on validation set
                #self.optimizer_G_dFF, self.optimizer_D_dFF, self.generator_dFF, self.discriminator_dFF, loss_dFF = train_GD(self.optimizer_G_dFF, self.optimizer_D_dFF, batch_dFF, self.generator_dFF, self.discriminator_dFF, self.patch, self.criterion_GAN, self.criterion_pixelwise, self.opt)
                #self.optimizer_G_SRO, self.optimizer_D_SRO, self.generator_SRO, self.discriminator_SRO, loss_SRO = train_GD(self.optimizer_G_SRO, self.optimizer_D_SRO, batch_SRO, self.generator_SRO, self.discriminator_SRO, self.patch, self.criterion_GAN, self.criterion_pixelwise, self.opt)
                # this is a little bit of cheating over discrim so may need to adjust discrim strength
                #self.optimizer_G_dFF, self.generator_dFF, self.optimizer_G_SRO, self.generator_SRO, loss_scat = train_both_scat(self.optimizer_G_dFF, batch_dFF, self.generator_dFF, self.optimizer_G_SRO, batch_SRO, self.generator_SRO, self.Scat_Loss, self.opt)

                sample_images(i, batch_dFF, batch_SRO, self.generator_dFF, self.generator_SRO, opt, self.image_loc)
        
################################################################################################################
####### MAIN RUN CODE #######

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--dataset_name", type=str, default="/data/lrudden/ML-DiffuseReader/dataset/validation/", help="validation name of the dataset")
    parser.add_argument("--experiment_name", type=str, default="first_test", help="name of expt")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
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

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    opt = parser.parse_args()

    T = Test(opt)
    T.test()
