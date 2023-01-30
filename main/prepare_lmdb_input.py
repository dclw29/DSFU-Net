"""
Create the LMDB dataset for DL input
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
        # 100 for testing
        return 205578 if train else 1000 # this number is specific to the number of TM examples we have for training
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)

def staple_image(image0, image1):
    """
    Based on two numpy array inputs (image0 and image1), stable the two together
    Ensure the target image is on the right, the conditioned image on the left
    """
    return np.concatenate((image0, image1), axis=1)
    
def read_single_lmdb(image_id=0, lmdb_dir="/data/lrudden/ML-DiffuseReader/dataset/training/train_lmdb_dFF"):
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
            lmdb_path = os.path.join(root, 'validation.lmdb')
        self.data_lmdb = lmdb.open(lmdb_path, readonly=True, max_readers=1,
                                   lock=False, readahead=False, meminit=False)
        self.is_encoded = is_encoded

    def __getitem__(self, index):

        with self.data_lmdb.begin(write=False, buffers=True) as txn:

            #data = txn.get(str(index).encode())
            data = txn.get(f"{index:08}".encode())

            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert('RGB') 
            else:
                img = pickle.loads(data)
                img = img.getimage()                

        if self.transform is not None:
            imgA, imgB = img 
            image_A = self.transform(imgA)
            image_B = self.transform(imgB)
        else:
            image_A, image_B = img 

        return {"A": image_A, "B": image_B}

    def __len__(self):
        return num_samples(self.name, self.train)


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

def store_many_lmdb(images, lmdb_dir="/home/lrudden/ML-DiffuseReader/dataset/training/train_lmdb"):
    """ Stores an array of images to LMDB.
        https://realpython.com/storing-images-in-python/#storing-to-lmdb
        Parameters:
        ---------------
        images       images array, (N, M, Mx2, 3) to be stored (2 x along width because of stapling)
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

def read_images(folder: str, tag: str, max_num: int) -> np.array:
    """
    Folder, conditional folder location
    tag is dFF or SRO
    max_num: number of images in folder (to loop through)
    """

    # read in order of images
    images = []
    # too much data, just use every 5th image for now
    for n in range(0, max_num+1, 5):
        image0 = np.array(Image.open(folder + "/Scattering/%i.png"%(n))) # conditional input
        image1 = np.array(Image.open(folder + "/" + tag + "/%i.png"%(n))) # the dFF or SRO are the target (from the scattering data)
        images.append(staple_image(image0, image1))
        if n % 92 == 0:
            print(">> %.2f completed"%(n * 100 /max_num))
    return np.asarray(images).astype(np.uint8)

def read_arrays(folder: str, tag: str) -> np.array:
    """
    Folder, conditional folder location of numpy array (post normalisation), it is the base array before Scattering or tag sep
    tag is dFF or SRO
    """
    # read in essentially random order (depending on what fs.encode decides to do)
    images = []
    afolder = os.fsencode(folder + "/Scattering")
    dir_files = [str(os.fsdecode(x).split("_scat.npy")[0]) for x in os.listdir(afolder) if os.fsdecode(x).endswith("scat.npy")]
    for f in dir_files:
        image0 = np.load(folder + "/Scattering/%s_scat.npy"%(f)) # conditional input
        image1 = np.load(folder + "/%s/%s_%s.npy"%(tag, f, tag)) # the dFF or SRO are the target (from the scattering data)
        images.append(staple_image(image0, image1))
    return np.concatenate(images, axis=2)

if __name__ == "__main__":

    # naming convention of input files
    # DiffuseScattering00001.npy, DiffuseScattering00002.npy... DiffuseScattering09999.npy and so on (5 digits)

    # load in each group and cat together

    trainingfolder = "/data/lrudden/ML-DiffuseReader/dataset/training/"

    images_dFF = read_arrays(trainingfolder + str(0), "dFF")
    images_SRO = read_arrays(trainingfolder + str(0), "SRO")
    groups = 5
    for g in range(1, groups):
        loadfolder = trainingfolder + str(g)
        images_dFF_tmp = read_arrays(loadfolder, "dFF")
        images_SRO_tmp = read_arrays(loadfolder, "SRO")
        images_dFF = np.concatenate((images_dFF, images_dFF_tmp), axis=2)
        images_SRO = np.concatenate((images_SRO, images_SRO_tmp), axis=2)
    store_many_lmdb(images_dFF, trainingfolder + "train_lmdb_dFF")
    store_many_lmdb(images_SRO, trainingfolder + "train_lmdb_SRO")

    # for testing
    #arr = read_single_lmdb()


