"""
Take slices of input training data to generate final 2D images we're going to train on within the network
Normalise over all inputs for final data
"""

import numpy as np
import torch
import sys, os
from geomloss import SamplesLoss
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image

def wasserstein_distance(X, Y):
    """
    Calculate W distance https://stackoverflow.com/questions/56820151/is-there-a-way-to-measure-the-distance-between-two-distributions-in-a-multidimen

    See here for nice explanation as to why it's better than KL: https://stats.stackexchange.com/questions/295617/what-is-the-advantages-of-wasserstein-metric-compared-to-kullback-leibler-diverg
    """
    Loss =  SamplesLoss("sinkhorn", blur=0.05,)
    return Loss( X, Y ).item()

def load_data(directory, filename, integer_leaps=16):
    """
    Load in the three relevent data for a scattering set and take slices based at integer leaps
    """
    scat_data = torch.tensor(np.load(directory + "/" + filename + ".npy"))
    SRO_data = torch.tensor(np.load(directory + "/" + filename[:-4] + "SRO.npy"))
    dFF_data = torch.tensor(np.load(directory + "/" + filename[:-6] + "dFF.npy"))
    return scat_data[:, :, ::16], SRO_data[:, :, ::16], dFF_data[:, :, ::16]

def register_unique_slices(data, slices, wasserstein_cutoff=5E-12):
    """
    See whether we have any unique data slices (i.e. cross-similarity between different returned matches of unique slices)
    slices is a list of indices to use
    """

    def _find_non_unique_(data, idx0, idx1):
        if wasserstein_distance(data[:,:,idx0], data[:,:,idx1]) < wasserstein_cutoff:
            return idx1
        else:
            return 0

    unique = [slices[0]]
    current_idx = 1
    while len(slices) > 1:
        delete_idx = _find_non_unique_(data, slices[0], slices[current_idx])
        if delete_idx != 0:
            slices = np.delete(slices, current_idx) # slices is now one value smaller (current_idx remains the same)
        else:
            unique.append(slices[current_idx]) # a rare occasion where we have a new value
            current_idx += 1

        if len(slices) == current_idx: # we've reached the end, reloop around
            current_idx = 1
            slices = np.delete(slices, 0)

    return np.asarray(unique)

def normalise_data_sum(data):
    """
    Normalise input data by the sum (prob distribution)
    """
    return (data + data.min().abs()) / torch.sum(data + data.min().abs())

def normalise_data_max(data):
    """
    Normalise input data by the maximum value
    """
    shift_val = data.min().abs()
    max_val = torch.max(data + data.min().abs())
    return (data + data.min().abs()) / torch.max(data + data.min().abs()), shift_val, max_val

def prep_dataset(directory, wasserstein_cutoff=5E-12, integer_leaps=16):
    """
    Loop through all numpy 3D arrays in folder and create larger array with each individual (correct) slice
    Choose what to extract here based on wasserstein metric (i.e. if greater than the cutoff, keep)
    Base wasserstein cutoff on random data? Or from difference between two completely different scattering sets...
    Should really be an average I think across lots of measured data
    # Consider every integer_leaps in scattering data 
    """
    folder = os.fsencode(directory)
    dir_files = [str(os.fsdecode(x).split(".")[0]) for x in os.listdir(directory) if os.fsdecode(x).endswith("Scat.npy")]
    
    # assuming 256 size input
    scat_data_all = torch.empty((256, 256, 1))
    SRO_data_all = torch.empty((256, 256, 1))
    dFF_data_all = torch.empty((256, 256, 1))
    for x, file in enumerate(sorted(dir_files)):
        scat_data, SRO_data, dFF_data = load_data(directory, file, integer_leaps)
        scat_data_norm = normalise_data_sum(scat_data) # temp norm for wasserstein, overall norm needs to be defined by dataset as a whole? 
        #TODO ask chloe about norm 

        # differences caused by taking pre-slices out from data
        # create wasserstein distance grid
        wass_dist = []
        for i in range(integer_leaps):
            for j in range(integer_leaps):
                wass_dist.append(wasserstein_distance(scat_data_norm[:,:,i], scat_data_norm[:,:,j]))
        wass_dist = np.triu(np.asarray(wass_dist).reshape(16, 16)) # only need upper triangle of data
        accepted = np.unique(np.where(wass_dist > wasserstein_cutoff))
        if len(accepted) == 0: # if there are no good distances, just take the middle slice
            unique = 8 # assuming 256 size split into lots of 15
            scat_data_all = torch.cat((scat_data_all, scat_data[:,:,unique][:,:,None]), axis=2)
            SRO_data_all = torch.cat((SRO_data_all, SRO_data[:,:,unique][:,:,None]), axis=2)
            dFF_data_all = torch.cat((dFF_data_all, dFF_data[:,:,unique][:,:,None]), axis=2)
        else:
            unique = register_unique_slices(scat_data_norm, accepted)
            # use unique to take final slice out
            scat_data_all = torch.cat((scat_data_all, scat_data[:,:,unique]), axis=2)
            SRO_data_all = torch.cat((SRO_data_all, SRO_data[:,:,unique]), axis=2)
            dFF_data_all = torch.cat((dFF_data_all, dFF_data[:,:,unique]), axis=2)
    
    return scat_data_all[:,:,1:], SRO_data_all[:,:,1:], dFF_data_all[:,:,1:]

def grey_plot(data, win="greymap", title="img"):
    """
    The original colourscale plot
    Adapted from visualise
    """

    data = data * 255 # we could store more information in colourmap

    # now to plot
    fig = plt.figure(figsize=(16, 16), dpi=16)
    plt.imshow(data, cmap='gray', vmin=0, vmax=255)

    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('tight')
    plt.axis('off')

    fig.savefig(title + ".png")
    plt.close(fig)

def colour_plot(data, title):
    """
    Adapted from visualise, upgrade from grey plot in an attempt to better train on colour images
    Try and prevent (mode collapse?) with certain pixels oversampling max value
    """

    data = data * 255 * 3 # times 3 for colours

    new_data = []
    for i in data.ravel():
        if i <= 255:
            new_data.append([i, 0., 0.])
        elif i > 255 and i <= 510:
            new_data.append([255, i-255, 0.])
        else:
            new_data.append([255, 255, i - 2*255])
    new_data = np.reshape(np.asarray(new_data), np.shape(data) + (3,)).astype(int)

    # now to plot
    fig = plt.figure(figsize=(16, 16), dpi=16) # produces images of 256 x 256 size - so each "pixel" now represented by 4x4 space
    plt.imshow(new_data, vmin=0, vmax=255)

    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('tight')
    plt.axis('off')

    fig.savefig(title + ".png")
    plt.close(fig)

if __name__ == "__main__":
    readfolder = "/home/lrudden/ML-DiffuseReader/dataset/raw_files/"
    savefolder = "/home/lrudden/ML-DiffuseReader/dataset/training/" # then we have dFF Scattering SRO (target is dFF and SRO)
    # dFF * SRO gives the scattering target

    scat_data, SRO_data, dFF_data = prep_dataset(readfolder)
    #TODO: something worth checking is the self wasserstein distance between this set

    # now normalise for each tensor, we need the terms to unnormalise later
    scat_data_norm, scat_shift_val, scat_max_val = normalise_data_max(scat_data)
    SRO_data_norm, SRO_shift_val, SRO_max_val = normalise_data_max(SRO_data)
    dFF_data_norm, dFF_shift_val, dFF_max_val = normalise_data_max(dFF_data)

    # save normalisation parameters
    np.save(savefolder + "scat_shift_val.npy", scat_shift_val); np.save(savefolder + "SRO_shift_val.npy", SRO_shift_val); np.save(savefolder + "dFF_shift_val.npy", dFF_shift_val)
    np.save(savefolder + "scat_max_val.npy", scat_max_val); np.save(savefolder + "SRO_max_val.npy", SRO_max_val); np.save(savefolder + "dFF_max_val.npy", dFF_max_val)
    # save tensors of big arrays
    np.save(savefolder + "scat_data_norm.npy", scat_data_norm); np.save(savefolder + "SRO_data_norm.npy", SRO_data_norm); np.save(savefolder + "dFF_data_norm.npy", dFF_data_norm)

    # load in
    scat_data_norm = torch.tensor(np.load(savefolder + "scat_data_norm.npy"))
    SRO_data_norm = torch.tensor(np.load(savefolder + "SRO_data_norm.npy"))
    dFF_data_norm = torch.tensor(np.load(savefolder + "dFF_data_norm.npy"))
    # now save data as images
    for cnt_d in range(scat_data_norm.size()[-1]):
        # Scattering
        colour_plot(scat_data_norm[:,:,cnt_d], title=savefolder + "Scattering/img_%04d"%(cnt_d))
        Image.open(savefolder + "Scattering/img_%04d.png"%(cnt_d)).convert('RGB').save(savefolder + "Scattering/img_%04d.png"%(cnt_d))
        # SRO
        colour_plot(SRO_data_norm[:,:,cnt_d], title=savefolder + "SRO/img_%04d"%(cnt_d))
        Image.open(savefolder + "SRO/img_%04d.png"%(cnt_d)).convert('RGB').save(savefolder + "SRO/img_%04d.png"%(cnt_d))
        # dFF (technically 66% redundant but easier to manage this way)
        colour_plot(dFF_data_norm[:,:,cnt_d], title=savefolder + "dFF/img_%04d"%(cnt_d))
        Image.open(savefolder + "dFF/img_%04d.png"%(cnt_d)).convert('RGB').save(savefolder + "dFF/img_%04d.png"%(cnt_d))