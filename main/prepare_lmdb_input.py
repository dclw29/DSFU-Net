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

def num_samples(dataset, train):
    if dataset == "scattering":
        # 1 for testing
        return 1 if train else 100 # this number is specific to the number of TM examples we have for training
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
        self.transform = transform
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
                img = img.convert('L') # L from RGB to deal with grayscale
            else:
                #### Original ##
                #img = np.asarray(data, dtype=np.uint8)
                # assume data is RGB
                #size = int(np.sqrt(len(img) / 3))
                #img = np.reshape(img, (size, size, 3))
                #img = Image.fromarray(img, mode='RGB')

                ### LSPR ###
                img = pickle.loads(data)
                # colour
                #size = int(np.sqrt(len(img) / 3))
                #img = np.reshape(img, (size, size, 3))
                #img = Image.fromarray(img, mode='RGB')
                img = img.getimage()                
                # grayscale
                #size = int(np.sqrt(len(img) / 1))
                #img = np.reshape(img, (size, size, 1))
                #img = Image.fromarray(img, mode='L')
                #img = lmdb_image.get_image()

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return num_samples(self.name, self.train)


class LMDB_Image:
    def __init__(self, image, mode="train"):

        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        #self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()

        #TODO need test information also

    def getimage(self):

        image = np.frombuffer(self.image, dtype=np.uint8)
        image = image.reshape(*self.size)
        h, w = self.size 
        image_A = image[:, : int(w / 2)]
        image_B = image[:, int(w / 2) :]
        #image_A = image.crop((0, 0, w / 2, h)) # note w/2 here, in other words, target image needs to be on the right
        #image_B = image.crop((w / 2, 0, w, h))  
        image_A = Image.fromarray(image_A).convert("RGB") #, mode="RGB")
        image_B = Image.fromarray(image_B).convert("RGB") #, mode="RGB")

        return {"A": image_A, "B": image_B}

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
    for n in range(max_num):
        image0 = np.array(Image.open(folder + "/" + tag + "/img_%04d.png"%(n)))
        image1 = np.array(Image.open(folder + "/Scattering/img_%04d.png"%(n)))
        images.append(staple_image(image0, image1))
    return np.asarray(images).astype(np.uint8)

if __name__ == "__main__":

    # naming convention of input files
    # DiffuseScattering00001.npy, DiffuseScattering00002.npy... DiffuseScattering09999.npy and so on (5 digits)

    trainingfolder = "/home/dclw/ML-DiffuseReader/dataset/training/"
    max_num = 271

    images_dFF = read_images(trainingfolder, "dFF", max_num)
    images_SRO = read_images(trainingfolder, "SRO", max_num)

    store_many_lmdb(images_dFF, trainingfolder + "train_lmdb_dFF")
    store_many_lmdb(images_SRO, trainingfolder + "train_lmdb_SRO")

    # for testing
    #arr = read_single_lmdb()


