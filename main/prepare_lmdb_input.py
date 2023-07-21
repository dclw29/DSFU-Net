"""
Create the LMDB dataset for DL input
Staple the non artefact data onto site c
"""

import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import lmdb
from PIL import Image
import pickle
import torchvision.transforms as transforms

def num_samples(dataset, train):
    if dataset == "scattering":
        return 198421 if train else 1000 # expanded dataset with symm data
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)

def staple_image(image0, image1, image2):
    """
    Based on two numpy array inputs (image0 and image1), stable the two together
    Ensure the target image is on the right, the conditioned image on the left
    """
    return np.concatenate((image0, image1, image2), axis=1)
    
def read_single_lmdb(image_id=0, lmdb_dir="/path/to/training/data/dataset/training/train_lmdb_dFF"):
    """ Stores a single image to LMDB.
        For testing purposes, return a single image from lmdb
        Parameters:
        ---------------
        image_id    integer unique ID for image
        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
    """

    # Open the LMDB environment
    env = lmdb.open(lmdb_dir, readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Encode the key the same way as we stored it
        data = txn.get(f"{image_id:08}".encode("ascii"))
        # Remember it's a CIFAR_Image object that is loaded
        lmdb_image = pickle.loads(data)
        # Retrieve the relevant bits
        image = lmdb_image.getimage()

    env.close()
    return image

class LMDBDataset(data.Dataset):
    def __init__(self, root, name='', train=True, transform=None, is_encoded=False):
        self.train = train
        self.name = name
        self.transform = transforms.Compose(transform)
        if self.train:
            lmdb_path = os.path.join(root)
        else:
            lmdb_path = os.path.join(root)#, 'validation.lmdb')
        self.data_lmdb = lmdb.open(lmdb_path, readonly=True, max_readers=1,
                                   lock=False, readahead=False, meminit=False)
        self.is_encoded = is_encoded

    def __getitem__(self, index):

        with self.data_lmdb.begin(write=False, buffers=True) as txn:

            data = txn.get(f"{index:08}".encode())

            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert('RGB') 
            else:
                img = pickle.loads(data)
                img = img.getimage()                

        if self.transform is not None:
            imgA, imgB, imgC = img 
            image_A = self.transform(np.array(imgA))
            image_B = self.transform(np.array(imgB))
            image_C = self.transform(np.array(imgC))
        else:
            image_A, image_B, image_C = img 
        return {"A": image_A, "B": image_B, "C" : image_C}

    def __len__(self):
        return num_samples(self.name, self.train)


class LMDB_Image:
    def __init__(self, image, mode="train"):

        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.size = image.shape
        self.image = image.tobytes()

    def getimage(self):

        image = self.image #np.frombuffer(self.image, dtype=np.float32)
        image = image.reshape(self.size)
        h, w = self.size 
        image_A = image[:, : int(w / 3)]
        image_B = image[:, int(w / 3) : 2*int(w / 3)]
        image_C = image[:, 2*int(w / 3):]

        return image_A, image_B, image_C 

def store_many_lmdb(images, lmdb_dir="/path/to/training/data/dataset/training/train_lmdb"):
    """ Stores an array of images to LMDB.
        https://realpython.com/storing-images-in-python/#storing-to-lmdb
        Parameters:
        ---------------
        images       images array, (N, H, Wx2) to be stored (2 x along width because of stapling)
    """
    num_images = len(images)

    map_size = num_images * images[0].nbytes * 10

    # Create a new LMDB DB for all the images
    env = lmdb.open(str(lmdb_dir), map_size=map_size)

    # Same as before — but let's write all the images in a single transaction
    with env.begin(write=True) as txn:
        for i in range(num_images):
            # All key-value pairs need to be Strings
            value = LMDB_Image(images[i])
            key = f"{i:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()

def normalise_data(data):
    """
    Normalise the data between 0 and 255 (for heatmap)
    """
    max_val = data.max()
    min_val = data.min()

    if min_val < 0:
        data = (data + np.abs(min_val)) / (max_val - min_val)
    else:
        data = (data - np.abs(min_val)) / (max_val - min_val)

    return data * 255

def read_arrays(folder: str, tag: str, get_metadata=False) -> np.array:
    """
    Folder, conditional folder location of numpy array (post normalisation), it is the base array before Scattering or tag sep
    tag is dFF or SRO
    """

    # create metadata mass folder if requested
    if get_metadata:
        molcode = []
        a = []; b = []; c = []
        corr = []

    # read in essentially random order (depending on what fs.encode decides to do)
    images = []
    afolder = os.fsencode(folder + "/Scattering")
    dir_files = [str(os.fsdecode(x).split("_scat.npy")[0]) for x in os.listdir(afolder) if os.fsdecode(x).endswith("scat.npy")]
    for f in dir_files:
        image0 = np.load(folder + "/Scattering/%s_scat.npy"%(f)) # conditional input
        image1 = np.load(folder + "/%s/%s_%s.npy"%(tag, f, tag)) # the dFF or SRO are the target (from the scattering data)
        image2 = np.load(folder + "/Scattering/%s_scat_clean.npy"%(f)) # non artefact data
        images.append(staple_image(image0, image1, image2))

        if get_metadata:
            molcode.append(np.load(folder + "/metadata/%s_molcode.npy"%(f)))
            a.append(np.load(folder + "/metadata/%s_a.npy"%(f)))
            b.append(np.load(folder + "/metadata/%s_b.npy"%(f)))
            c.append(np.load(folder + "/metadata/%s_c.npy"%(f)))
            corr.append(np.load(folder + "/metadata/%s_corr.npy"%(f)))

    if get_metadata:
        return np.concatenate(images, axis=2), dir_files, np.concatenate(molcode, axis=0), np.concatenate(a, axis=0), np.concatenate(b, axis=0), np.concatenate(c, axis=0), np.concatenate(corr, axis=0)
    else:
        return np.concatenate(images, axis=2), dir_files

if __name__ == "__main__":

    # naming convention of input files
    # DiffuseScattering00001.npy, DiffuseScattering00002.npy... DiffuseScattering09999.npy and so on (5 digits)

    # load in each group and cat together
    trainingfolder = "/path/to/training/data/dataset/training/"
    #trainingfolder = "/path/to/validation/data/validation/" # validationfolder

    images_dFF, dir_files = read_arrays(trainingfolder + str(0), "dFF")
    groups = 4 #  5 for training
    for g in range(1, groups):
        loadfolder = trainingfolder + str(g)
        images_dFF_tmp, dir_files_tmp, molcode, a, b, c, corr = read_arrays(loadfolder, "dFF", get_metadata=True)
        images_dFF = np.concatenate((images_dFF, images_dFF_tmp), axis=2)
        dir_files = dir_files  + dir_files_tmp
    images_dFF = np.swapaxes(np.swapaxes(images_dFF, -1, 0), 1, 2)
    #store_many_lmdb(images_dFF, trainingfolder + "val_lmdb_dFF")
    store_many_lmdb(images_dFF, trainingfolder + "train_lmdb_dFF")

    np.save(trainingfolder + "molcodes.npy", molcode) # this should be in the correct order that train dff is saved
    np.save(trainingfolder + "conc_a.npy", a); np.save(trainingfolder + "conc_b.npy", b); np.save(trainingfolder + "conc_c.npy", c)
    np.save(trainingfolder + "correlations.npy", corr)

    del images_dFF
    del images_dFF_tmp

    images_SRO, _ = read_arrays(trainingfolder + str(0), "SRO")
    for g in range(1, groups):
        loadfolder = trainingfolder + str(g)
        images_SRO_tmp, _ = read_arrays(loadfolder, "SRO")
        images_SRO = np.concatenate((images_SRO, images_SRO_tmp), axis=2)
    images_SRO = np.swapaxes(np.swapaxes(images_SRO, -1, 0), 1, 2)
    #store_many_lmdb(images_SRO, trainingfolder + "val_lmdb_SRO")
    store_many_lmdb(images_SRO, trainingfolder + "train_lmdb_SRO")

    # for testing
    #arr = read_single_lmdb()


