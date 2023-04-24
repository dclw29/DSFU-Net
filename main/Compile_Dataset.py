"""
Take slices of input training data to generate final 2D images we're going to train on within the network
Normalise over all inputs for final data
"""

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import sys, os
from geomloss import SamplesLoss

def artefact_creation(A, data):
    """
    Create the artefact (i.e. white lines) on the scattering data to mirror experimental results
    """
    return A * data

def define_wasser_cutoff(no_checks=20000):
    """
    Calculate what the wasser cutoff should be based on moving between two sum normalised random dist
    Check 20000 for good statistics
    """
    W = []
    for d in range(no_checks):
        data = torch.rand(256, 256, 2)
        data_norm = normalise_data_sum(data)
        W.append(wasserstein_distance(data_norm[:,:,0], data_norm[:,:,1]))
    return np.mean(np.abs(W))
    
def wasserstein_distance(X, Y):
    """
    Calculate W distance https://stackoverflow.com/questions/56820151/is-there-a-way-to-measure-the-distance-between-two-distributions-in-a-multidimen

    See here for nice explanation as to why it's better than KL: https://stats.stackexchange.com/questions/295617/what-is-the-advantages-of-wasserstein-metric-compared-to-kullback-leibler-diverg
    """
    Loss =  SamplesLoss("sinkhorn", blur=0.05,)
    return Loss( X, Y ).item()

def load_data(directory, filename):
    """
    Load in the three relevent data for a scattering set and take slices based at integer leaps
    #CHANGED no longer take integer leaps, take all values as we now only return 12 sensible suggestions
    """
    scat_data = torch.tensor(np.load(directory + "/" + filename + ".npy"))
    SRO_data = torch.tensor(np.load(directory + "/" + filename[:-4] + "SRO.npy"))
    dFF_data = torch.tensor(np.load(directory + "/" + filename[:-4] + "dFF.npy"))
    # load meta data
    molcode = np.load(directory + "/" + filename[:-4] + "molcode_metadata.npy")
    qmax = np.load(directory + "/" + filename[:-4] + "qmax_metadata.npy")
    m1 = np.load(directory + "/" + filename[:-4] + "m1_metadata.npy")
    s = np.load(directory + "/" + filename[:-4] + "s_metadata.npy")
    #a = np.load(directory + "/" + filename[:-4] + "a_metadata.npy")
    #b = np.load(directory + "/" + filename[:-4] + "b_metadata.npy")
    #c = np.load(directory + "/" + filename[:-4] + "c_metadata.npy")
    #corr = np.load(directory + "/" + filename[:-4] + "corr_metadata.npy")
    #return scat_data, SRO_data, dFF_data, molcode, a, b, c, corr
    return scat_data, SRO_data, dFF_data, molcode, qmax, m1, s 

def register_unique_slices(data, slices, wasserstein_cutoff=6.7E-11):
    """
    See whether we have any unique data slices (i.e. cross-similarity between different returned matches of unique slices)
    slices is a list of indices to use
    cutoff is calculated from moving between two random distributions (the mean of 10000 wasserstein distances)
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
            unique.append(slices[current_idx]) # we have a unique image to add
            current_idx += 1
            # we now need to check if anything left in slices is unique w.r.t this image 
            # delete if so, keep otherwise (so we don't accidentally add another non-unique image)
            if len(slices) > 2:
                # slices changes dynamically, use a shift to avoid over counting beyond the bounds
                shift = 0
                for x in range(len(slices)-current_idx):
                    delete_idx2 = _find_non_unique_(data, slices[current_idx-1], slices[current_idx+x-shift])
                    if delete_idx2 != 0:
                        slices = np.delete(slices, current_idx+x-shift)
                        shift += 1
                    else:
                        continue

        if len(slices) == current_idx: # we've reached the end, reloop around
            current_idx = 1
            slices = np.delete(slices, 0)

    return np.unique(unique)

def normalise_data_sum(data):
    """
    Normalise input data by the sum (prob distribution)
    """
    assert data.min() >= 0, "Input data for normalisation contains negative values!"
    min_data = data.view(256*256, -1).min(dim=0)[0]
    data -= min_data.unsqueeze(0).unsqueeze(0)
    sum_data = data.view(256*256,-1).sum(dim=0)
    return data / sum_data.unsqueeze(0).unsqueeze(0)

def normalise_data_max(data, buffer=0.01):
    """
    Normalise input data by the maximum value
    Buffer is how much we set as the min value such that any scattering data isn't lost during normalisation
    data input is size (256x256xn) where n is no. of accepted examples after wasserstein check
    """
    assert data.min() >= 0, "Input data for normalisation contains negative values!"
    min_data = data.view(256*256, -1).min(dim=0)[0]
    data -= min_data.unsqueeze(0).unsqueeze(0)
    max_data = data.view(256*256, -1).max(dim=0)[0]
    data /= max_data.unsqueeze(0).unsqueeze(0)
    # shift space between 1-buffer and buffer so we don't lose scattering data
    return (data * (1-buffer)) + buffer

def randomise_arty(art, no_images=144):
    """
    art should have a size of 256x256x256x6.
    We want 144 256x256 images as output to use on our data. Choose a random slice from the 4th axis first
    (different set of artefacts), then select first from x, then y, then z a random slice using a random no.
    from a normal dist.
    """
    slices = torch.empty(256, 256, 1)
    loop = int(no_images / 3) # for each axis
    for s in range(loop): # loop through x, y then z
        select = np.random.randint(6)
        # use a normal dist to select where (so more likely in the centre)
        dim_select = np.random.normal(loc=0.5, scale=0.152)
        if dim_select > 1: dim_select = 1
        elif dim_select < 0: dim_select = 0
        dim_select = int(dim_select * 255)
        slices = torch.cat((slices, art[dim_select, :, :, select].unsqueeze(-1)), dim=2)
    # now do the same for y and z
    for s in range(loop): # loop through x, y then z
        select = np.random.randint(6)
        # use a normal dist to select where (so more likely in the centre)
        dim_select = np.random.normal(loc=0.5, scale=0.152)
        if dim_select > 1: dim_select = 1
        elif dim_select < 0: dim_select = 0
        dim_select = int(dim_select * 255)
        slices = torch.cat((slices, art[:, dim_select, :, select].unsqueeze(-1)), dim=2)    
    for s in range(loop): # loop through x, y then z
        select = np.random.randint(6)
        # use a normal dist to select where (so more likely in the centre)
        dim_select = np.random.normal(loc=0.5, scale=0.152)
        if dim_select > 1: dim_select = 1
        elif dim_select < 0: dim_select = 0
        dim_select = int(dim_select * 255)
        slices = torch.cat((slices, art[:, :, dim_select, select].unsqueeze(-1)), dim=2)        

    # shuffle output 
    idx = torch.randperm(no_images)
    slices = slices[:,:,1:]
    return slices[:,:,idx]

def prep_dataset(directory, molcode, artefact_folder="/home/lrudden/ML-DiffuseReader/Artefacts/", wasserstein_cutoff=6.7E-11):
    """
    Loop through all numpy 3D arrays in folder and create larger array with each individual (correct) slice
    Choose what to extract here based on wasserstein metric (i.e. if greater than the cutoff, keep)
    Base wasserstein cutoff on random data? Or from difference between two completely different scattering sets...
    
    We have 12 sets of concentrations generated, each with 12 images
    So we need 144 sets of artefacts, decided randomly but along the three dimensions
    so 48 artefacts along each dimension (decided randomly)
    
    molcode: which set of arrays for a molecule am I reading in?

    """

    # preload all artefacts and split into units of 16
    arty = torch.empty(256,256,256,1)
    afolder = os.fsencode(artefact_folder)
    afiles = [str(os.fsdecode(x)) for x in os.listdir(afolder) if os.fsdecode(x).endswith(".npy")]
    # Take 12 random artefact slices 
    for afile in afiles:
        arty = torch.cat((arty, torch.tensor(np.load(artefact_folder + afile)).unsqueeze(-1)), dim=3)
    arty = arty[:, :, :, 1:]
    # now randomly select 144 slices from the arty array to apply to our images down the line
    arty = randomise_arty(arty)

    folder = os.fsencode(directory)
    dir_files = [str(os.fsdecode(x).split(".npy")[0]) for x in os.listdir(directory) if os.fsdecode(x).endswith("scat.npy") and molcode in os.fsdecode(x)]
    
    # assuming 256 size input
    scat_data_all = torch.empty((256, 256, 1))
    SRO_data_all = torch.empty((256, 256, 1))
    dFF_data_all = torch.empty((256, 256, 1))
    #molcode_all = []; a_all = []; b_all = []; c_all = []; corr_all = []
    molcode_all = []; qmax_all = []; s_all = []; m1_all = []
    for x, file in enumerate(sorted(dir_files)):
        #scat_data, SRO_data, dFF_data, molcode, a, b, c, corr = load_data(directory, file)
        scat_data, SRO_data, dFF_data, molcode, qmax, m1, s = load_data(directory, file)

        # Apply artefacts at the END, i.e. don't include then in wasserstein checks

        # check the wasserstein distance between scat data. If the same, discard
        # Otherwise, save and combine with artefact at very end
        scat_data_norm = normalise_data_sum(scat_data) # temp norm for wasserstein 

        # differences caused by taking pre-slices out from data
        # create wasserstein distance grid
        wass_dist = []
        for i in range(scat_data_norm.size()[-1]):   #TODO: cheapen this by only running on top triangle of data
            for j in range(scat_data_norm.size()[-1]):
                if j < i:
                    wass_dist.append(0) # could put as part of for loop but want wass_dist to be matrix
                else:
                    wass_dist.append(wasserstein_distance(scat_data_norm[:,:,i], scat_data_norm[:,:,j]))        
        wass_dist = np.triu(np.asarray(wass_dist).reshape(scat_data_norm.size()[-1], scat_data_norm.size()[-1])) # only need upper triangle of data
        accepted = np.unique(np.where(wass_dist > wasserstein_cutoff)) # only take when there are big differences
        if len(accepted) == 0: # if there are no good distances, just take the middle slice
            unique = int(scat_data_norm.size()[-1] / 2) # assuming 256 size split into lots of 15
            scat_data_all = torch.cat((scat_data_all, scat_data[:,:,unique][:,:,None]), axis=2)
            SRO_data_all = torch.cat((SRO_data_all, SRO_data[:,:,unique][:,:,None]), axis=2)
            dFF_data_all = torch.cat((dFF_data_all, dFF_data[:,:,unique][:,:,None]), axis=2)
        else:
            unique = register_unique_slices(scat_data_norm, accepted)
            # use unique to take final slice out
            scat_data_all = torch.cat((scat_data_all, scat_data[:,:,unique]), axis=2)
            SRO_data_all = torch.cat((SRO_data_all, SRO_data[:,:,unique]), axis=2)
            dFF_data_all = torch.cat((dFF_data_all, dFF_data[:,:,unique]), axis=2)   

            # apply unique to metadata also
            molcode_all.append(molcode[unique])
            #a_all.append(a[unique]); b_all.append(b[unique]); c_all.append(c[unique])
            #corr_all.append(corr[unique])
            qmax_all.append(qmax[unique])
            m1_all.append(m1[unique])
            s_all.append(s[unique])

    molcode_all = np.concatenate(molcode_all)
    #a_all = np.concatenate(a_all); b_all = np.concatenate(b_all); c_all = np.concatenate(c_all)
    #corr_all = np.concatenate(corr_all)
    qmax_all = np.concatenate(qmax_all); m1_all = np.concatenate(m1_all); s_all = np.concatenate(s_all)

    scat_data_all = scat_data_all[:,:,1:]
    SRO_data_all = SRO_data_all[:,:,1:]
    dFF_data_all = dFF_data_all[:,:,1:]

    # now apply our artifacts to the accepted images
    final_accepted_size = scat_data_all.size()[-1]
    arty = arty[:,:,:final_accepted_size]

    # normalise our data between 0.01 and 1 (plus some buffer to account for minimum scattering not being zero)
    # Each slice needs to be normalised by itself
    #scat_data_all = normalise_data_max(scat_data_all)
    #SRO_data_all = normalise_data_max(SRO_data_all)
    #dFF_data_all = normalise_data_max(dFF_data_all)

    # try normalising between 0 and 1 instead (trying to get SRO to work)
    scat_data_all = normalise_data_max(scat_data_all, buffer=0)
    SRO_data_all = normalise_data_max(SRO_data_all, buffer=0)
    dFF_data_all = normalise_data_max(dFF_data_all, buffer=0)

    # Apply artifact to scattering data
    scat_data_art = artefact_creation(arty, scat_data_all)
        
    #return scat_data_all, SRO_data_all, dFF_data_all, scat_data_art, molcode_all, a_all, b_all, c_all, corr_all
    return scat_data_all, SRO_data_all, dFF_data_all, scat_data_art, molcode_all, qmax_all, m1_all, s_all

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

def normalize_one_to_one(D):
    """
    data should be of size 256 x 256
    return normalize between 0 and 1 (will become 0-255 later)
    """
    data = D.clone()
    data_min = data.min()
    if data.min() < 0:
        data += data.min().abs()
    else:
        data -= data.min()
    data_max = data.max()
    data /= data.max()
    return data, data_min, data_max #(data * 2) - 1
 
def main(molcode, readfolder="/home/lrudden/ML-DiffuseReader/dataset/raw_files/", savefolder="/home/lrudden/ML-DiffuseReader/dataset/training/", artifactfolder="/home/lrudden/ML-DiffuseReader/Artefacts/"):
    """
    count is the current image integer to differentiate between images (not related to cnt_d)
    """
    # dFF * SRO gives the scattering target

    # prep our data to create images (internal wasserstein checks are performed)
    #scat_data, SRO_data, dFF_data, scat_data_art, molcode_all, a, b, c, corr = prep_dataset(readfolder, molcode, artefact_folder=artifactfolder)
    scat_data, SRO_data, dFF_data, scat_data_art, molcode_all, qmax_all, m1_all, s_all = prep_dataset(readfolder, molcode, artefact_folder=artifactfolder)
    # They should come out normalised (between buffer and 1) and with artefacts applied to scattering data
    np.save(savefolder + "dFF/" + molcode + "_" + "dFF.npy", dFF_data.numpy().astype(np.float32))
    np.save(savefolder + "SRO/" + molcode + "_" + "SRO.npy", SRO_data.numpy().astype(np.float32))
    np.save(savefolder + "Scattering/" + molcode + "_" + "scat.npy", scat_data_art.numpy().astype(np.float32))
    np.save(savefolder + "Scattering/" + molcode + "_" + "scat_clean.npy", scat_data.numpy().astype(np.float32))

    # save smaller metadata

    #np.save(savefolder + "metadata/" + molcode + "_" + "molcode.npy", molcode_all)
    #np.save(savefolder + "metadata/" + molcode + "_" + "a.npy", a)
    #np.save(savefolder + "metadata/" + molcode + "_" + "b.npy", b)
    #np.save(savefolder + "metadata/" + molcode + "_" + "c.npy", c)
    #np.save(savefolder + "metadata/" + molcode + "_" + "corr.npy", corr)
    np.save(savefolder + "metadata/" + molcode + "_" + "molcode.npy", molcode_all)
    np.save(savefolder + "metadata/" + molcode + "_" + "qmax.npy", qmax_all)
    np.save(savefolder + "metadata/" + molcode + "_" + "m1.npy", m1_all)
    np.save(savefolder + "metadata/" + molcode + "_" + "s.npy", s_all)

    # save normalised output in another folder

    """
    # now save data as images
    for cnt_d in range(scat_data.size()[-1]):
        # Scattering
        colour_plot(scat_data[:,:,cnt_d], title=savefolder + "Scattering/img_%06d"%(count))
        Image.open(savefolder + "Scattering/img_%06d.png"%(count)).convert('RGB').save(savefolder + "Scattering/img_%06d.png"%(count))
        # SRO
        colour_plot(SRO_data[:,:,cnt_d], title=savefolder + "SRO/img_%06d"%(count))
        Image.open(savefolder + "SRO/img_%06d.png"%(count)).convert('RGB').save(savefolder + "SRO/img_%06d.png"%(count))
        # dFF (technically 66% redundant but easier to manage this way)
        colour_plot(dFF_data[:,:,cnt_d], title=savefolder + "dFF/img_%06d"%(count))
        Image.open(savefolder + "dFF/img_%06d.png"%(count)).convert('RGB').save(savefolder + "dFF/img_%06d.png"%(count))
        count += 1
    """
