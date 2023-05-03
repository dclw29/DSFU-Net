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
#from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from math import sqrt

import matplotlib.pyplot as plt
# load in user image (in np format)
# normalise between -1 and 1
# Run generator on it
# save output as numpy array between 0 and 1

# add optional function that loops through folder if requested

def normalise(data):
    """
    Normalise data to between -1 and 1 for network input
    """
    # in case input data has nan or -1 (for artefacts), which won't be in actual scattering input, set to zero first.
    data[data == -1] = 0
    data[np.isnan(data)] = 0

    return ((data / data.max()) * 2) - 1

def read_and_save(filename):
    """
    Read a filename, generate new data, and save
    """

    data = np.load(filename)
    
    # for now, reorder input data to be of the right shape
    #data = data.swapaxes(0, -1)
    #data = data.swapaxes(1, 2)
    
    data = torch.tensor(normalise(data), dtype=torch.float32).unsqueeze(0) # add channel dimension
    data = inverse_spread(data)
    sample_dFF = generator_dFF(data).detach().cpu().numpy()
    sample_SRO = generator_SRO(data).detach().cpu().numpy()

    outname = filename.split(".")[0]

    # move the samples between 0 and 1
    sample_dFF = (sample_dFF + 1) / 2
    sample_SRO = (sample_SRO + 1) / 2
    np.save(outname + "_dFF.npy", sample_dFF)
    np.save(outname + "_SRO.npy", sample_SRO)
 
    plt.imshow(sample_dFF[0, 0])
    plt.savefig("dFF_output.png")
    plt.imshow(sample_SRO[0, 0])
    plt.savefig("SRO_output.png")

def inverse_spread(data):
    """
    Lots of values are grouped up around -1 (or < -0.95), spread them out
    """

    min_shift_val=0.
    max_shift_val=2**0.5
    #shifted_data = torch.clamp((data + 1 + torch.rand(*data.shape)/1000)**0.1, max=1)
    data = data + 1
    shifted_data = data.sqrt()
    min_shift_val = shifted_data.min() # keep record to unnormalise later, max will still be 1
    renorm_data_tmp = shifted_data - min_shift_val
    max_shift_val = renorm_data_tmp.max()
    renorm_data = (renorm_data_tmp / max_shift_val) * 2 - 1
    return renorm_data

######### READ ARGUMENTS ########

parser = argparse.ArgumentParser()
# default will be final value we put on github, but in theory people can train for longer, use their own model etc.
parser.add_argument("--epoch", type=int, default=12, help="Epoch to read generator model from")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--filename", type=str, default="default.npy", help="Filename of input numpy scattering data to be converted. Not read if using folder mode")
parser.add_argument("--folder", type=str, default="", help="If specified location, pipeline will instead read and convert all npy arrays in this folder. Will crash if data is not scattering input")
parser.add_argument("--generator_loc", type=str, default="../models/", help="Specify the location of the generator pytorch models to load in")

opt = parser.parse_args()

# read input arguments
filename = opt.filename
folder = opt.folder
epoch = opt.epoch
img_height = opt.img_height
img_width = opt.img_width
channels = opt.channels
generator_loc = opt.generator_loc

device = torch.device("cpu")

# Initialize generators
#generator_dFF = GeneratorUNet_Attn(in_channels=channels, out_channels=channels)
#generator_SRO = GeneratorUNet_Attn(in_channels=channels, out_channels=channels)
generator_dFF = GeneratorUNet(in_channels=channels, out_channels=channels).to(device)
generator_SRO = GeneratorUNet(in_channels=channels, out_channels=channels).to(device)

# Load a pre-existing network
generator_dFF.load_state_dict(torch.load("%s/generator_dFF_%d.pth" % (generator_loc, epoch)))
generator_SRO.load_state_dict(torch.load("%s/generator_SRO_%d.pth" % (generator_loc, epoch)))
#TODO check normalisation in normal input (0.5 input tensor) vs what you are doing - why even do it, we could apply our own normalisation prior to input

# load in data
if len(folder) == 0:
    read_and_save(filename)
else:
    read_folder = os.fsencode(folder)
    files = [str(os.fsdecode(x)) for x in os.listdir(read_folder) if os.fsdecode(x).endswith(".npy")]
    for filename in files:
        read_and_save(filename)
