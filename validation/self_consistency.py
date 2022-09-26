"""
Check whether slices from 3D data are equivilent, or whether we can use them as separate data samples
"""

import numpy as np
import sys, os
import re
from geomloss import SamplesLoss
import random as rand
rand.seed()
import torch
import matplotlib.pyplot as plt

def wasserstein_distance(X, Y):
    """
    Calculate W distance https://stackoverflow.com/questions/56820151/is-there-a-way-to-measure-the-distance-between-two-distributions-in-a-multidimen

    See here for nice explanation as to why it's better than KL: https://stats.stackexchange.com/questions/295617/what-is-the-advantages-of-wasserstein-metric-compared-to-kullback-leibler-diverg
    """
    Loss =  SamplesLoss("sinkhorn", blur=0.05,)
    return Loss( X, Y ).item()

def wasserstein_self_check(data, slice1, slice2):
    """
    Are slice1 and slice2, taken from the same 3D generated output, identical?
    Check between 3D generated outputs (different molecules), and from same to get an idea of the spread
    """

def wasserstein_random(size):
    """
    What is the wasserstein distance between two completely normalised uniformaly noisy datasets of size size?
    """
    d1 = np.random.uniform(size=size)
    d2 = np.random.uniform(size=size)

    # normalise by mass
    d1 /= np.sum(d1)
    d2 /= np.sum(d2)

    return wasserstein_distance(torch.tensor(d1), torch.tensor(d2))

def wasserstein_non_self_check(data1, data2)


if __name__ == "__main__":
    scat_data_Al_tol = torch.tensor(np.load("AlBCToluene_1_0_Scat.npy"))
    # wasserstein distance needs mass not max (i.e. sum must equal 1)
    scat_norm_Al_tol = (scat_data_Al_tol + scat_data_Al_tol.min().abs()) / torch.sum(scat_data_Al_tol + scat_data_Al_tol.min().abs())

    dist_AL_tol = []
    for i in range(0, 256, 16):
        for j in range(0, 256, 16):
            dist_AL_tol.append(wasserstein_distance(scat_norm_Al_tol[:,:,i], scat_norm_Al_tol[:,:,j]))
    dist_AL_tol = np.asarray(dist_AL_tol).reshape(16,16)
    plt.plot(np.arange(256), dist_AL_tol)
    plt.xlabel("Slice from normalised scattering data")
    plt.ylabel("Wassterstein Distance")
    plt.title("AL-Toluene")
    plt.savefig("AL_toluene_self_wass_dist.png")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(scat_norm_Al_tol[:,:,15], cmap="gray") # index minus 1 remember
    ax2.imshow(scat_norm_Al_tol[:,:,255], cmap="gray")
    plt.show()

    # plot the heatmaps
    plt.imshow(scat_norm_Al_tol[:,:,64], cmap="gray")
    #plt.savefig("AL_toluene_slice0.png")
    plt.show()
    plt.imshow(scat_norm_Al_tol[:,:,128], cmap="gray")
    #plt.savefig("AL_toluene_slice8.png")
    plt.show()   

    # Compare also between two different molecules
    scat_data_tol_tol = torch.tensor(np.load("BCToluene_1BCToluene_2_0_Scat.npy"))
    scat_norm_tol_tol = (scat_data_tol_tol + scat_data_tol_tol.min().abs()) / torch.sum(scat_data_tol_tol + scat_data_tol_tol.min().abs())
    
    dist_tol_tol = []
    for i in range(0, 256, 16):
        for j in range(0, 256, 16):
            dist_tol_tol.append(wasserstein_distance(scat_norm_tol_tol[:,:,i], scat_norm_tol_tol[:,:,j]))
    dist_tol_tol = np.asarray(dist_tol_tol).reshape(16,16)

    plt.plot(np.arange(256), dist_tol_tol)
    plt.xlabel("Slice from normalised scattering data")
    plt.ylabel("Wassterstein Distance")
    plt.title("toluene-Toluene")
    plt.savefig("toluene_toluene_self_wass_dist.png")
    plt.show()    

    # plot the heatmaps
    plt.imshow(scat_norm_tol_tol[:,:,0], cmap="gray")
    plt.savefig("toluene_toluene_slice0.png")
    plt.show()
    plt.imshow(scat_norm_tol_tol[:,:,14], cmap="gray")
    plt.savefig("toluene_toluene_slice14.png")
    plt.show()   

    # compare dist from different sources
    dist_AlTol_TolTol = []
    for i in range(0,256,16):
        for j in range(0,256,16):
            dist_AlTol_TolTol.append(wasserstein_distance(scat_norm_Al_tol[:,:,i], scat_norm_tol_tol[:,:,j]))
    dist_AlTol_TolTol = np.asarray(dist_AlTol_TolTol).reshape(16,16)

    plt.plot(np.arange(256), dist_AlTol_TolTol)
    plt.xlabel("Slice from normalised scattering data")
    plt.ylabel("Wassterstein Distance")
    plt.title("Al-Toluene Toluene-Toluene")
    plt.savefig("Al_toluene_to_toluene_toluene_dist_wass_dist")
    plt.show()       
