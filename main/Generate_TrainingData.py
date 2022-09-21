'''
Create training data to separate diffuse scattering
'''

import sys, os
import re
import numpy as np
import matplotlib.pyplot as plt
from meta_library import elements, numbers, atomic_SF, molecules
import time
from itertools import combinations, permutations
import random as rand
rand.seed()

###### Initialise grid to generate data from #####

t0 = time.time()
print("Creating q-grid")
# Create Q grid
a = 8
b = 8
c = 8
hklMax = 8
resolution = 256
gridSize = (2*hklMax)/resolution
#hklVec = np.zeros((resolution, resolution, resolution,3))
#QMag = np.zeros((resolution, resolution, resolution))
#print(np.shape(hklVec[0,0,:,0]))

# Do maths
hklVec = np.mgrid[0:resolution, 0:resolution, 0:resolution].T
hklVec = np.flip(hklVec*gridSize - hklMax, axis=3)
QMag = 2*np.pi*np.sqrt(np.sqrt((hklVec[:,:,:,0]/a)**2+(hklVec[:,:,:,1]/b)**2+(hklVec[:,:,:,2]/c)**2))

### old ###
#for i in range(resolution):
#    for j in range(resolution):
#        for k in range(resolution):
#            hklVec[i,j,k,:] = np.array([-hklMax+(i*gridSize),-hklMax+(j*gridSize),-hklMax+(k*gridSize)])
#QMag = 2*np.pi*np.sqrt(np.sqrt((hklVec[:,:,:,0]/a)**2+(hklVec[:,:,:,1]/b)**2+(hklVec[:,:,:,2]/c)**2))

# Read in molecules
mols = molecules.keys()
a = list(combinations(mols,2))
molComb = np.asarray([list(x) for x in list(a)])

for i in range(len(molComb)): # can we do things outside this loop?
    mol1 = molecules[molComb[i,0]]
    mol2 = molecules[molComb[i,1]]
    moleculeCode = molComb[i,0] + molComb[i,1]
    print("Molecule: " + moleculeCode)
    
    # Calculate atomic scattering factors for relevant atoms
    # Save them as dictionaries so they only need calculating once
    # this is faster than using np unique
    atomtypes = []
    for a in range(len(mol1)):
        if numbers[int(mol1[a,0])] not in atomtypes: atomtypes.append(numbers[int(mol1[a,0])])
    for a in range(len(mol2)):
        if numbers[int(mol2[a,0])] not in atomtypes: atomtypes.append(numbers[int(mol2[a,0])])

    print("Calculating atomic form factors")
    atomFormFactors = {}
    Qmag_4pi = QMag/(4*np.pi)
    for a in range(len(atomtypes)):
        aFF = np.zeros((resolution, resolution, resolution))
        at = atomic_SF[atomtypes[a]]
        #TODO factorise
        for i in range(0,4): #TODO does this need to be hardcoded?
            #aFF += at[f'a{i+1}'] * np.exp(-(at[f'b{i+1}'] * (QMag/(4*np.pi))**2)) # add a1-4 and b1-4 in different stages
            aFF += at[f'a{i+1}'] * np.exp(-(at[f'b{i+1}'] * (Qmag_4pi)**2)) # since these are preset, we could save all of this data into prebuilt arrays (in dict)
        aFF += at['c']
    
        atomFormFactors[atomtypes[a]] = aFF

    # new
    print("Calculating molecular form factors")
    molFormFactor1 = np.zeros((resolution,resolution,resolution)).astype(complex)
    molFormFactor2 = np.zeros((resolution,resolution,resolution)).astype(complex)
    # einsum is way quicker than dot product and we can calculate on mass multiplication
    mol1_matrix = np.exp(-2j * np.pi * np.einsum("ijkl,ol->oijk", hklVec, mol1[:, 1:]))
    mol2_matrix = np.exp(-2j * np.pi * np.einsum("ijkl,ol->oijk", hklVec, mol2[:, 1:])) #TODO: Check if this could be wrong because of zeros...?
    # create molFormFactor arrays
    for a in range(len(mol1)): # can we use eisum here somehow?
        molFormFactor1 += atomFormFactors[numbers[int(mol1[a,0])]] * mol1_matrix[a]
    for a in range(len(mol2)):
        molFormFactor2 += atomFormFactors[numbers[int(mol2[a,0])]] * mol2_matrix[a]

    dFF = np.transpose((abs(molFormFactor1-molFormFactor2)**2))

    #### old ####
    # Calculate molecular form factors
    #t2 = time.time()
    #molFormFactor1 = np.zeros((resolution,resolution,resolution)).astype(complex)
    #molFormFactor2 = np.zeros((resolution,resolution,resolution)).astype(complex)
    #for a in range(len(mol1)): # mol2 doesn't matter? Is this an error?
    #    molFormFactor1 += atomFormFactors[numbers[int(mol1[a,0])]]* np.exp(-2j * np.pi * (np.dot(hklVec, mol1[a,1:])))
    #for a in range(len(mol2)):
    #    molFormFactor2 += atomFormFactors[numbers[int(mol2[a,0])]]* np.exp(-2j * np.pi * (np.dot(hklVec, mol2[a,1:])))
    #
    #dFF = np.transpose((abs(molFormFactor1-molFormFactor2)**2))
    #t3 = time.time()
    #plt.imshow(dFF[128,:,:])
    #plt.show() 

    np.save(moleculeCode+"_dFF.npy", dFF)
    #plt.imshow(dFF[128,:,:])
    #plt.savefig(moleculeCode+"_dFF.png")
    
    #### new ####
    print("Calculating short range order")
    # these definitions can be put outside the loop
    v = np.asarray([[0,0,1],[1,1,0],[1,0,0],[0,1,1],[0,1,0],[1,0,1],[1,1,1]])
    concentrations = np.arange(0.1,0.6,0.1)
    M2 = 1 - concentrations
    # prep dot product values
    SRO_cos = np.cos(2 * np.pi * np.einsum("ijkl,ol->oijk", hklVec, v))
    for c in range(len(concentrations)):
        print(" Concentration:" + str(concentrations[c]))
        p = [rand.uniform(0, concentrations[c]) for i in range(len(v))]
        al = np.zeros(len(p))
        for i in range(len(p)):
            al[i] = 1-(p[i]/(concentrations[c] * M2[c]))      
        
        SRO = 2 * np.sum(al[:,None,None,None] * SRO_cos, axis=0).T + 1
        np.save(moleculeCode + "_" + str(c) + "_SRO.npy", SRO)
        # Image testing
        plt.imshow(SRO[128,:,:])
        plt.savefig(moleculeCode+"_"+str(c)+"_SRO.png")

        scat = dFF*SRO
        np.save(moleculeCode+"_"+str(c)+"_Scat.npy", scat)
        plt.imshow(scat[128,:,:])
        plt.savefig(moleculeCode+"_"+str(c)+"_scat.png")
        
    ### old ####
    #v = np.asarray([[0,0,1],[1,1,0],[1,0,0],[0,1,1],[0,1,0],[1,0,1],[1,1,1]])
    #concentrations = np.arange(0.1,0.6,0.1)
    #for c in range(len(concentrations)):
    #    print(" Concentration:" + str(concentrations[c]))
    #    m1 = concentrations[c]
    #    m2 = 1 - m1
    #    p = [rand.uniform(0,concentrations[c]) for i in range(len(v))]
    #    al = np.zeros(len(p))
    #    for i in range(len(p)):
    #        al[i] = 1-(p[i]/(m1*m2))
    #
    #
    #    SRO = np.zeros((resolution,resolution,resolution))
    #    for i in range(len(v)):
    #        SRO += al[i]*np.cos(2*np.pi*np.dot(hklVec, v[i,:]))
    #
    #    SRO = np.transpose(SRO*2 + 1)
    #    #np.save(moleculeCode+"_"+str(c)+"_SRO.npy", SRO)
    #    #plt.imshow(SRO[128,:,:])
    #    #plt.savefig(moleculeCode+"_"+str(c)+"_SRO.png")
    #
    #    scat = dFF*SRO
    #    #np.save(moleculeCode+"_"+str(c)+"_Scat.npy", scat)
    #    #plt.imshow(scat[128,:,:])
    #    #plt.savefig(moleculeCode+"_"+str(c)+"_scat.png")

    t1 = time.time()
    
print(" Time taken: {:.4f}s".format(t1-t0))