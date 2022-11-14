'''
Create training data to separate diffuse scattering
'''

import sys, os
sys.path.append("/data/lrudden/ML-DiffuseReader/main")
import Compile_Dataset # for the main function to make images. We can then delete the source arrays (saves space)
import re
import numpy as np
import matplotlib.pyplot as plt
from meta_library import elements, numbers, atomic_SF, molecules
import time
from itertools import combinations, permutations
import random as rand
rand.seed()
import logging

# to generate data on slurm
start = sys.argv[1] # start and end point to loop through molComb
end = sys.argv[2]

# create logfile
logger = logging.getLogger("log_%s_gen_data"%(str(start)))
fh = logging.FileHandler("/data/lrudden/ML-DiffuseReader/main/logfiles/log_%s.log"%str(start))
ch = logging.StreamHandler()
logger.addHandler(fh)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

#TODO: Apply rototranslations to input data (use grid of space in internal coordinates)

###### Initialise grid to generate data from #####

savefolder = "/data/lrudden/ML-DiffuseReader/dataset/raw_files/%s/"%str(start)
imagefolder = "/data/lrudden/ML-DiffuseReader/dataset/training/%s/"%str(start)
# create necessary folders
os.mkdir(savefolder)
os.mkdir(imagefolder)
for s in ["Scattering", "SRO", "dFF"]:
    os.mkdir(imagefolder + s)

t0 = time.time()
logger.info("> Creating q-grid")
# Create Q grid
a = 8
b = 8
c = 8
hklMax = 8
resolution = 256
gridSize = (2*hklMax)/resolution

# Do maths
hklVec = np.mgrid[0:resolution, 0:resolution, 0:resolution].T
hklVec = np.flip(hklVec*gridSize - hklMax, axis=3)
QMag = 2*np.pi*np.sqrt(np.sqrt((hklVec[:,:,:,0]/a)**2+(hklVec[:,:,:,1]/b)**2+(hklVec[:,:,:,2]/c)**2))

# Read in molecules
mols = molecules.keys()
a = list(combinations(mols,2))
molComb = np.asarray([list(x) for x in list(a)])
count = 0

#for i in range(len(molComb)):
for i in range(int(start), int(end)):
    t1_0 = time.time()

    mol1 = molecules[molComb[i,0]]
    mol2 = molecules[molComb[i,1]]
    moleculeCode = molComb[i,0] + molComb[i,1]
    logger.info("> Using Molecule: %s"%str(moleculeCode))
    
    # Calculate atomic scattering factors for relevant atoms
    # Save them as dictionaries so they only need calculating once
    # this is faster than using np unique
    atomtypes = []
    for a in range(len(mol1)):
        if numbers[int(mol1[a,0])] not in atomtypes: atomtypes.append(numbers[int(mol1[a,0])])
    for a in range(len(mol2)):
        if numbers[int(mol2[a,0])] not in atomtypes: atomtypes.append(numbers[int(mol2[a,0])])

    logger.info("> Calculating atomic form factors")
    atomFormFactors = {}
    Qmag_4pi = QMag/(4*np.pi)
    for a in range(len(atomtypes)):
        aFF = np.zeros((resolution, resolution, resolution))
        at = atomic_SF[atomtypes[a]]

        for i in range(0,4): #TODO does this need to be hardcoded?
            aFF += at[f'a{i+1}'] * np.exp(-(at[f'b{i+1}'] * (Qmag_4pi)**2)) # since these are preset, we could save all of this data into prebuilt arrays (in dict)
        aFF += at['c']
        atomFormFactors[atomtypes[a]] = aFF

    logger.info("> Calculating molecular form factors")
    molFormFactor1 = np.zeros((resolution,resolution,resolution)).astype(complex)
    molFormFactor2 = np.zeros((resolution,resolution,resolution)).astype(complex)
    # einsum is way quicker than dot product and we can calculate on mass multiplication
    mol1_matrix = np.exp(-2j * np.pi * np.einsum("ijkl,ol->oijk", hklVec, mol1[:, 1:]))
    mol2_matrix = np.exp(-2j * np.pi * np.einsum("ijkl,ol->oijk", hklVec, mol2[:, 1:])) 
    # create molFormFactor arrays
    for a in range(len(mol1)): # can we use eisum here somehow?
        molFormFactor1 += atomFormFactors[numbers[int(mol1[a,0])]] * mol1_matrix[a]
    for a in range(len(mol2)):
        molFormFactor2 += atomFormFactors[numbers[int(mol2[a,0])]] * mol2_matrix[a]

    dFF = np.transpose((abs(molFormFactor1-molFormFactor2)**2))

    np.save(savefolder + moleculeCode+"_dFF.npy", dFF)
    #plt.imshow(dFF[128,:,:])
    #plt.savefig(moleculeCode+"_dFF.png")
    
    #### new ####
    logger.info("> Calculating short range order")
    # these definitions can be put outside the loop
    v = np.asarray([[0,0,1],[1,1,0],[1,0,0],[0,1,1],[0,1,0],[1,0,1],[1,1,1]])
    concentrations = np.arange(0.1,0.6,0.1)

    M2 = 1 - concentrations
    # prep dot product values
    SRO_cos = np.cos(2 * np.pi * np.einsum("ijkl,ol->oijk", hklVec, v))
    for c in range(len(concentrations)):
        logger.info("> Making Concentration: %s"%(str(concentrations[c])))
        p = [rand.uniform(0, concentrations[c]) for i in range(len(v))]
        al = np.zeros(len(p))
        for i in range(len(p)):
            al[i] = 1-(p[i]/(concentrations[c] * M2[c]))      
        
        SRO = 2 * np.sum(al[:,None,None,None] * SRO_cos, axis=0).T + 1
        scat = dFF*SRO
        np.save(savefolder + moleculeCode + "_" + str(c) + "_SRO.npy", SRO)
        # Image testing
        #plt.imshow(SRO[128,:,:])
        #plt.savefig(moleculeCode+"_"+str(c)+"_SRO.png")

        np.save(savefolder + moleculeCode+"_"+str(c)+"_Scat.npy", scat)
        #plt.imshow(scat[128,:,:])
        #plt.savefig(moleculeCode+"_"+str(c)+"_scat.png")
    
    old_count = count
    count = Compile_Dataset.main(count=count, readfolder=savefolder, savefolder=imagefolder, artifactfolder="/data/lrudden/ML-DiffuseReader/Artefacts/")
    # now delete the source arrays (saves hard disk space)
    folder = os.fsencode(savefolder)
    npy_files = [str(os.fsdecode(x).split(".")[0]) for x in os.listdir(folder) if os.fsdecode(x).endswith(".npy")]
    for f in npy_files:
        os.remove(savefolder + f + ".npy")
        
    t1_1 = time.time()
    logger.info("> Generated %i images in %.4fs"%(count-old_count, t1_1-t1_0))
    
    
logger.info("### Total Time taken: {:.4f}s ###".format(t1_1-t0))
