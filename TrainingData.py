import re
import numpy as np
import matplotlib.pyplot as plt
from _elements import elements_, numbers_
from scatFact import atoms
from Molecules import molecules_
import time
from itertools import combinations, permutations
import random as rand
rand.seed()

'''
Create training data to separate diffuse scattering
'''
t0 = time.time()
print("Creating q-grid")
# Create Q grid
a = 8
b = 8
c = 8
hklMax = 8
resolution = 256
gridSize = (2*hklMax)/resolution
hklVec = np.zeros((resolution,resolution,resolution,3))
QMag = np.zeros((resolution,resolution,resolution))
#print(np.shape(hklVec[0,0,:,0]))
for i in range(resolution):
    for j in range(resolution):
        for k in range(resolution):
            hklVec[i,j,k,:] = np.array([-hklMax+(i*gridSize),-hklMax+(j*gridSize),-hklMax+(k*gridSize)])
            
QMag = 2*np.pi*np.sqrt(np.sqrt((hklVec[:,:,:,0]/a)**2+(hklVec[:,:,:,1]/b)**2+(hklVec[:,:,:,2]/c)**2))
t1 = time.time()
print(" Time taken: {:.4f}s".format(t1-t0))

# Read in molecules
mols = molecules_.keys()
a = list(combinations(mols,2))
molComb = np.asarray([list(x) for x in list(a)])

for i in range(len(molComb)):
    mol1 = molecules_[molComb[i,0]]
    mol2 = molecules_[molComb[i,1]]
    moleculeCode = molComb[i,0]+molComb[i,1]
    print("Molecule:" + moleculeCode)
    atomtypes = []

    # Calculate atomic scattering factors for relevant atoms
    # Save them as dictionaries so they only need calculating once
    
    for a in range(len(mol1)):
        if numbers_[int(mol1[a,0])] not in atomtypes: atomtypes.append(numbers_[int(mol1[a,0])])
    for a in range(len(mol2)):
        if numbers_[int(mol2[a,0])] not in atomtypes: atomtypes.append(numbers_[int(mol2[a,0])])
    
    print("Calculating atomic form factors")
    atomFormFactors = {}
    for a in range(len(atomtypes)):
        aFF = np.zeros((resolution,resolution,resolution))
        for i in range(0,4):
            at = atoms[atomtypes[a]]
            aFF += at[f'a{i+1}']*np.exp(-(at[f'b{i+1}']*(QMag/(4*np.pi))**2))
        aFF += at['c']
    
        atomFormFactors[atomtypes[a]] = aFF
    #plt.imshow(atomFormFactors["C"][:,:,128])
    #plt.show()
    
    t2 = time.time()
    print(" Time taken: {:.4f}s".format(t2-t1))
    
    # Calculate molecular form factors
    print("Calculating molecular form factors")
    molFormFactor1 = np.zeros((resolution,resolution,resolution)).astype(complex)
    molFormFactor2 = np.zeros((resolution,resolution,resolution)).astype(complex)
    #print(len(mol1))
    for a in range(len(mol1)):
        molFormFactor1 += atomFormFactors[numbers_[int(mol1[a,0])]]*np.exp(-2j*np.pi*(np.dot(hklVec,mol1[a,1:])))
        molFormFactor2 += atomFormFactors[numbers_[int(mol1[a,0])]]*np.exp(-2j*np.pi*(np.dot(hklVec,mol2[a,1:])))
    
    dFF = np.transpose((abs(molFormFactor1-molFormFactor2)**2))
    #plt.imshow(dFF[128,:,:])
    #plt.show() 
    np.save(moleculeCode+"_dFF.npy", dFF)
    plt.imshow(dFF[128,:,:])
    plt.savefig(moleculeCode+"_dFF.png")
    
    t3 = time.time()
    print(" Time taken: {:.4f}s".format(t3-t2))
    
    print("Calculating short range order")
    v = np.asarray([[0,0,1],[1,1,0],[1,0,0],[0,1,1],[0,1,0],[1,0,1],[1,1,1]])
    concentrations = np.arange(0.1,0.6,0.1)
    for c in range(len(concentrations)):
        print(" Concentration:" + str(concentrations[c]))
        m1 = concentrations[c]
        m2 = 1-m1
        p = [rand.uniform(0,concentrations[c]) for i in range(len(v))]
        al = np.zeros(len(p))
        for i in range(len(p)):
            al[i] = 1-(p[i]/(m1*m2))
    
    
        SRO = np.zeros((resolution,resolution,resolution))
        for i in range(len(v)):
            SRO += al[i]*np.cos(2*np.pi*np.dot(hklVec,v[i,:]))
    
        SRO = np.transpose(SRO*2 + 1)
        np.save(moleculeCode+"_"+str(c)+"_SRO.npy", SRO)
        plt.imshow(SRO[128,:,:])
        plt.savefig(moleculeCode+"_"+str(c)+"_SRO.png")
    
        scat = dFF*SRO
        np.save(moleculeCode+"_"+str(c)+"_Scat.npy", scat)
        plt.imshow(scat[128,:,:])
        plt.savefig(moleculeCode+"_"+str(c)+"_scat.png")
    
    t4 = time.time()
    print(" Time taken: {:.4f}s".format(t4-t3))