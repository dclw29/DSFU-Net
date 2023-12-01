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
Pipeline to run our DSFU-Net on desired samples
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
from argparse import Namespace

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

"""
Process:
Load in user image (in np format)
Normalise between -1 and 1
Run generator on it
Save output as numpy array between 0 and 1

Optional function that loops through folder if requested
"""

def normalise(data, zero_shift=False):
    """
    Normalise data to between -1 and 1 for network input
    Input data should ideally contain no negative values, but that may not be true...
    :params data: raw data to input
    :params zero_shift: If False, all values less than 0 will be set to zero. If True, data is shifted forwards by the minimum value that is less than zero
    """
    # in case input data has nan (for artefacts) or is less than zero, which won't be in actual scattering input, set to zero first.
    data[np.isnan(data)] = 0
    if zero_shift:
        data += -data.min()
    else:
        data[data < 0] = 0

    return ((data / data.max()) * 2) - 1

def read_and_save(filename, device, no_norm=False, zero_shift=False):
    """
    Read a filename, generate new data, and save
    """

    data = np.load(filename)
    
    if not no_norm:
        data = torch.tensor(normalise(data, zero_shift), dtype=torch.float32).to(device) # add channel dimension
    else:
        data = torch.tensor(data, dtype=torch.float32).to(device)

    if data.shape == (256, 256):
        data = data.unsqueeze(0).unsqueeze(0)
    elif data.shape == (1, 256, 256):
        data = data.unsqueeze(0)
    elif data.shape == (1, 1, 256, 256):
        pass
    else:
        raise Exception("Data input shape of ", data.shape, " not recognised! Please reshape to 256x256")

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
    plt.savefig("%s_dFF_output.png"%(outname))
    plt.imshow(sample_SRO[0, 0])
    plt.savefig("%s_SRO_output.png"%(outname))

def inverse_spread(data):
    """
    Lots of values are grouped up around -1 (or < -0.95), spread them out
    """

    min_shift_val=0.
    max_shift_val=2**0.5
    data = data + 1
    shifted_data = data.sqrt()
    min_shift_val = shifted_data.min() # keep record to unnormalise later, max will still be 1
    renorm_data_tmp = shifted_data - min_shift_val
    max_shift_val = renorm_data_tmp.max()
    renorm_data = (renorm_data_tmp / max_shift_val) * 2 - 1
    return renorm_data

######### READ ARGUMENTS ########

parser = argparse.ArgumentParser()
# default will be final value we put on GitHub, but in theory people can train for longer, use their own model etc.
parser.add_argument("--epoch", type=int, default=200, help="Epoch to read generator model from. Default is 200 (corresponding to saved models).")
parser.add_argument("--img_height", type=int, default=256, help="size of image height (default is 256). Can only be changed if you retrain the network.")
parser.add_argument("--img_width", type=int, default=256, help="size of image width (default is 256). Can only be changed if you retrain the network.")
parser.add_argument("--channels", type=int, default=1, help="Number of channels in input image. (default is 1). Can only be changed if you retrain the network.")
parser.add_argument("--filename", type=str, default="default.npy", help="Filename of input numpy scattering data to be converted. Not read if using folder mode. Default is default.npy")
parser.add_argument("--folder", type=str, default="", help="If specified location, pipeline will instead read and convert all npy arrays in this folder. Will crash if data is not scattering input")
parser.add_argument("--generator_loc", type=str, default="%s/../models/"%curr_dir, help="Specify the location of the generator pytorch models to load in. Default is the ../models/ folder containing the saved models.")
parser.add_argument("--device", type=str, default="cpu", help="What device to run the code on? Default is cpu, but could replace with gpu:0 depending on your hardware.")
parser.add_argument("--no_norm", default=False, action='store_true', help="Don't normalise the input data between -1 and 1. Note, if you are running the demo (or have already normalised), use this!")
parser.add_argument("--zero_shift", default=False, action='store_true', help="Ideally there should be no negative values in your input. But if there is, when normalising, shift all values by the most negative upwards. If not included, all negative values are set to zero by default.")

opt = parser.parse_args()

# read input arguments
filename = opt.filename
folder = opt.folder
epoch = opt.epoch
img_height = opt.img_height
img_width = opt.img_width
channels = opt.channels
generator_loc = opt.generator_loc
gpu = opt.device
no_norm = opt.no_norm
zero_shift = opt.zero_shift

device = torch.device(gpu)

# Initialize generators
generator_dFF = GeneratorUNet(in_channels=channels, out_channels=channels).to(device)
generator_SRO = GeneratorUNet(in_channels=channels, out_channels=channels).to(device)

# Load a pre-existing network
generator_dFF.load_state_dict(torch.load("%s/generator_dFF_%d.pth" % (generator_loc, epoch), map_location=device))
generator_SRO.load_state_dict(torch.load("%s/generator_SRO_%d.pth" % (generator_loc, epoch), map_location=device))

# load in data
if len(folder) == 0:
    read_and_save(filename, device, no_norm, zero_shift)
else:
    read_folder = os.fsencode(folder)
    files = [str(os.fsdecode(x)) for x in os.listdir(read_folder) if os.fsdecode(x).endswith(".npy")]
    for filename in files:
        read_and_save(filename, device, no_norm)
