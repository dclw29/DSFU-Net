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
Create training data to train pix2pix gan network to separate diffuse scattering into short range and long range order
"""

import numpy as np
import sys, os
curr_dir = os.get_cwd()
sys.path.append(curr_dir)
sys.path.append("%s/../main/")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from meta_library_2 import elements, numbers, atomic_SF
from itertools import combinations, permutations
import math
import random as rand
import time
rand.seed()
import logging
import re 
import Compile_Dataset

TestSet = False

if TestSet:
    from meta_library import Testmolecules as molecules
else:
    from meta_library import molecules

def get_fractional_to_cartesian_matrix(a, b, c, alpha, beta, gamma,
                                       angle_in_degrees=True):
    """
    Return the transformation matrix that converts fractional coordinates to
    cartesian coordinates.
    Parameters
    ----------
    a, b, c : float
        The lengths of the edges.
    alpha, gamma, beta : float
        The angles between the sides.
    angle_in_degrees : bool
        True if alpha, beta and gamma are expressed in degrees.
    Returns
    -------
    r : array_like
        The 3x3 rotation matrix. ``V_cart = np.dot(r, V_frac)``.
    """
    if angle_in_degrees:
        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)
        gamma = np.deg2rad(gamma)
    cosa = np.cos(alpha)
    sina = np.sin(alpha)
    cosb = np.cos(beta)
    sinb = np.sin(beta)
    cosg = np.cos(gamma)
    sing = np.sin(gamma)
    volume = 1.0 - cosa**2.0 - cosb**2.0 - cosg**2.0 + 2.0 * cosa * cosb * cosg
    volume = np.sqrt(volume)
    r = np.zeros((3, 3))
    r[0, 0] = a
    r[0, 1] = b * cosg
    r[0, 2] = c * cosb
    r[1, 1] = b * sing
    r[1, 2] = c * (cosa - cosb * cosg) / sing
    r[2, 2] = c * volume / sing
    return r

def calculate_scattering(mol1,mol2,QVec,QMag):
    # Store the atomic form factors in a dictionary so they only need calculating once
    atomtypes = []
    coords1 = mol1['coords'].values[0]
    coords2 = mol2['coords'].values[0]
    for a in range(len(coords1)):
        if numbers[int(coords1[a,0])] not in atomtypes: atomtypes.append(numbers[int(coords1[a,0])])
    for a in range(len(coords2)):
        if numbers[int(coords2[a,0])] not in atomtypes: atomtypes.append(numbers[int(coords2[a,0])])

    #form factors = sum(1,4) [a*exp(-b*(q/4pi)^2)] + c
    atomFormFactors = {}
    for a in range(len(atomtypes)):
        aFF = np.zeros((np.shape(QMag)))
        at = atomic_SF[atomtypes[a]]
        for i in range(0,4):
            aFF += at[f'a{i+1}'] * np.exp(-(at[f'b{i+1}'] * (Qmag/(4*np.pi))**2)) 
        aFF += at['c']
        atomFormFactors[atomtypes[a]] = aFF

    # Calculate molecular form factors
    molFormFactor1 = np.zeros((np.shape(QMag))).astype(complex)
    molFormFactor2 = np.zeros((np.shape(QMag))).astype(complex)

    # no 2pi if using Q and cartesian space 
    mol1_matrix = np.exp(-1j * np.einsum("ijkl,ol->oijk", QVec, coords1[:, 1:]))
    mol2_matrix = np.exp(-1j * np.einsum("ijkl,ol->oijk", QVec, coords2[:, 1:]))

    for i in range(len(coords1)):
        molFormFactor1 += atomFormFactors[numbers[int(coords1[i,0])]]*mol1_matrix[i]
    for i in range(len(coords2)):
        molFormFactor2 += atomFormFactors[numbers[int(coords2[i,0])]]*mol2_matrix[i]

    # dFF = difference in molecular form factors
    dFF = np.zeros((np.shape(QMag)))
    dFF = m1*m2*(abs(molFormFactor1-molFormFactor2)**2)

    # Calculate short range order
    SRO = np.zeros((np.shape(QMag)))
    for i in range(len(IAVectors)): 
        vec = np.dot(rot, IAVectors[i,:])
        SRO_cos = np.cos(np.einsum("ijkl,l->ijk", QVec, vec))
        SRO += 2*corr[i]*SRO_cos
                
    # Calculate total scattering
    SRO = np.where(SRO<0,0,SRO)
    scat = dFF*(SRO)

    return dFF, SRO, scat

def make_q_grids(size,gridSize,Qmax,recip_length,shift=0):
    QVec = np.mgrid[0:size[0],0:size[1],0:size[2]]
    QVec = np.moveaxis(QVec, 0, -1)

    for i in range(3):
        if i == np.argmin(size): QVec[:,:,:,i] = (QVec[:,:,:,i]*recip_length)+shift
        else: QVec[:,:,:,i] = QVec[:,:,:,i]*gridSize - Qmax
    
    Qmag = np.sqrt((QVec[:,:,:,0])**2+(QVec[:,:,:,1])**2+(QVec[:,:,:,2])**2)

    return QVec, Qmag

############################
# Initialisation Parameters
# what is the current folder? (ideally what is the current directory this file is in)
folder = curr_dir

correlations = np.load(folder + "/CorrelationGeneration/FINAL/alphas.npy")
concs = np.load(folder + "/CorrelationGeneration/FINAL/concentrations.npy")
IAVectors = np.load(folder + "/CorrelationGeneration/FINAL/IJK_vectors.npy")

Qmax = 8.0
resolution = 256.0
gridSize = (2*Qmax)/resolution
############################

# To speed things up (and make things parralellisable) we need to not generate all data at once. 
# So only load molecules corresponding to input group, then split further
input_group = int(sys.argv[1])
counter = int(sys.argv[2]) # where do we start counting concentrations alphas from in the big list?
# alter with parrelisation

# create logfile
# Where will we save the data?
if TestSet:
    logger = logging.getLogger("log_%s_gen_data_test"%(str(input_group)))
    fh = logging.FileHandler("%s/logfiles/log_%s_validation.log"%(folder,str(input_group)))
    savefolder = "%s/../dataset/raw_files/validation/%s/"%(folder,str(input_group))
    imagefolder = "%s/../dataset/validation/%s/"%(folder,str(input_group))
else:
    logger = logging.getLogger("log_%s_gen_data"%(str(input_group)))
    fh = logging.FileHandler("%s/logfiles/log_%s.log"%(folder,str(input_group)))
    savefolder = "%s/../dataset/raw_files/%s/"%(folder,str(input_group))
    imagefolder = "%s/../dataset/training/%s/"%(folder,str(input_group))

ch = logging.StreamHandler()
logger.addHandler(fh)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

# create necessary folders
if not os.path.exists(savefolder):
    os.mkdir(savefolder)
if not os.path.exists(imagefolder):
    os.mkdir(imagefolder)
for s in ["Scattering", "SRO", "dFF", "metadata"]:
    if not os.path.exists(imagefolder + s):
        os.mkdir(imagefolder + s)

logger.info(">>>> Beginning generation for group %s <<<<"%input_group)

# get the list of molecule names
names = list(molecules.query('group == @input_group')['name'])

# get the unique combinations of molecules
a = list(combinations(names,2))
molComb = np.asarray([list(x) for x in list(a)])

# we want to save molcode, concentration, correlations

# extract info for the pair
for pair in range(len(molComb)): #len(molComb)
    n1 = molComb[pair,0]
    n2 = molComb[pair,1]
    molcode = str(n1)+str(n2)
    
    t0 = time.time()

    logger.info(">> Doing molecules "+n1+" and "+n2)
    idx1 = molecules[molecules['name']==str(n1)].index
    idx2 = molecules[molecules['name']==str(n2)].index
    mol1 = molecules.iloc[idx1]        
    mol2 = molecules.iloc[idx2]

    # loop over multiple correlations 
    for conc_counter in range(12): # 992 combinations in the dataset. ~12 different correlation values per molecule pair
        m1 = concs[counter]
        m2 = 1-m1
        corr = correlations[counter,:]
        counter += 1
        logger.info(">> Beginning Concentration %.2f"%(m1))

        # Re-calculate cell parameters based on concentrations
        # angles stay the same
        a = (mol1['lps'].values[0][0]*m2)+(mol2['lps'].values[0][0]*m1)
        b = (mol1['lps'].values[0][1]*m2)+(mol2['lps'].values[0][1]*m1)
        c = (mol1['lps'].values[0][2]*m2)+(mol2['lps'].values[0][2]*m1)

        logger.info(">> Recalculated cell parameters based on concentrations: a=%.2f, b=%.2f, c=%.2f"%(a,b,c))
        alpha = 90.0
        gamma = 90.0

        if mol1['group'].values[0] == 3: beta = 120.0
        else: beta = 90.0

        # Calculate reciprocal lattice
        rot = get_fractional_to_cartesian_matrix(a, b, c, alpha, beta, gamma,True)
        aVec = np.dot(rot,np.array([1,0,0]))
        bVec = np.dot(rot,np.array([0,1,0]))
        cVec = np.dot(rot,np.array([0,0,1]))
        astar = 2*np.pi*np.cross(bVec,cVec)/(np.dot(aVec,np.cross(bVec,cVec)))
        bstar = 2*np.pi*np.cross(cVec,aVec)/(np.dot(bVec,np.cross(cVec,aVec)))
        cstar = 2*np.pi*np.cross(aVec,bVec)/(np.dot(cVec,np.cross(aVec,bVec)))

        # Get main correlations - permutations of 1 0 0 
        cx = corr[121]
        cy = corr[11]
        cz = corr[1]

        logger.info(">> Main correlations are: cx=%.2f, cy=%.2f, cz=%.2f"%(cx,cy,cz))

        # Select hkl planes to calculate
        # if the correlation along y axis is positive, the scattering maxima will be at Bragg positions
        # which we can calculate from the unit cell dimensions
        if cy > -0.05:
            # calculate h0l, h1l and h2l planes
            # Create Q grid
            # set Qmax as 8 - just less than Qmax on BM0l at D = 200, lam = 0.7
            QVec,Qmag = make_q_grids([resolution,4,resolution],gridSize,Qmax,bstar[1])
            
            # calculate h0l, h1l, h2l and h3l planes
            dFF_uxw, SRO_uxw, scat_uxw = calculate_scattering(mol1,mol2,QVec,Qmag)
            ind = 0

        # this time we dont know where the scattering maxima will be so we can calculate a series of planes
        # between h0l and h1l and take the most intense
        elif cy < 0:
            QVec = np.mgrid[0:resolution,0:int(1/gridSize),0:resolution]
            QVec = np.moveaxis(QVec, 0, -1)
            QVec[:,:,:,0] = QVec[:,:,:,0]*gridSize - Qmax
            QVec[:,:,:,1] = (QVec[:,:,:,1]*gridSize)
            QVec[:,:,:,2] = QVec[:,:,:,2]*gridSize - Qmax
    
            Qmag = np.sqrt((QVec[:,:,:,0])**2+(QVec[:,:,:,1])**2+(QVec[:,:,:,2])**2)

            dFF_uxw, SRO_uxw, scat_uxw = calculate_scattering(mol1,mol2,QVec,Qmag)

            #for i in range(int(1/gridSize)):
            #    plt.imshow(scat_uxw[:,i,:].T,cmap = "hot", origin = "lower")
            #    plt.colorbar()
            #    plt.show()
            #    plt.clf()

            # find the value of x with most scattering
            summed = np.sum(scat_uxw,axis=(0,2))
            maxScat = np.argmax(summed)
            #plt.plot(summed)
            #plt.show()
            #plt.clf()

            # get corresponding QVec
            q_shift = maxScat*gridSize
            QVec,Qmag = make_q_grids([resolution,4,resolution],gridSize,Qmax,bstar[1],q_shift)
            # recalculate scattering on new grid
            dFF_uxw, SRO_uxw, scat_uxw = calculate_scattering(mol1,mol2,QVec,Qmag)
            ind = 0

            #for i in range(4):
            #    plt.imshow(scat_uxw[:,i,:].T,cmap = "hot", origin = "lower")
            #    plt.colorbar()
            #    plt.show()
            #    plt.clf()
            
        # Find the xkl and hkx planes with the most scattering, using h1l as base (excludes possiblilty of h0l systematic absences)
        # For hkx, sum along x axis. Because planes should be symmetric, only need to do half. -10 so that full peak at 0 can be seen        
        x = np.sum(scat_uxw[:,ind,int(resolution/2)-10:], axis = 0)
        z = np.sum(scat_uxw[int(resolution/2)-10:,ind,:], axis = -1)

        # Find the peaks in this list
        peaks_x, details_x = find_peaks(x, prominence=0.75)
        heights_x = details_x['prominences']
        peaks_z, details_z = find_peaks(z, prominence=0.75)
        heights_z = details_z['prominences']
            
        # Use the highest peaks to find the indices of l
        # If fewer than 4 peaks were found, add extra random planes     
        highest_scatter_x = peaks_x[np.argpartition(heights_x, -min(len(peaks_x),4))[-min(len(peaks_x),4):]]-10+int(resolution/2)
        highest_scatter_z = peaks_z[np.argpartition(heights_z, -min(len(peaks_z),4))[-min(len(peaks_z),4):]]-10+int(resolution/2)

        if len(peaks_x) < 4: 
            plane_x = np.random.randint(int(resolution/2),high=int(resolution),size = 4-len(peaks_x))
            highest_scatter_x = np.append(highest_scatter_x,plane_x)
        if len(peaks_z) < 4: 
            plane_z = np.random.randint(int(resolution/2),high=int(resolution),size = 4-len(peaks_z))
            highest_scatter_z = np.append(highest_scatter_z,plane_z)

        # Find the corresponding QVec
        q_uvx = highest_scatter_x*gridSize - Qmax
        q_xvw = highest_scatter_z*gridSize - Qmax

        # calculate new scattering planes
        QVec = np.mgrid[0:resolution,0:resolution,0:4]
        QVec = np.moveaxis(QVec,0,-1)
        QVec[:,:,:,2] = q_uvx
        QVec[:,:,:,0] = QVec[:,:,:,0]*gridSize - Qmax
        QVec[:,:,:,1] = QVec[:,:,:,1]*gridSize - Qmax 
        Qmag = np.sqrt((QVec[:,:,:,0])**2+(QVec[:,:,:,1])**2+(QVec[:,:,:,2])**2)
        dFF_uvx, SRO_uvx, scat_uvx = calculate_scattering(mol1,mol2,QVec,Qmag)

        QVec = np.mgrid[0:4,0:resolution,0:resolution]
        QVec = np.moveaxis(QVec,0,-1)
        QVec[:,:,:,0] = q_xvw[:,None,None]
        QVec[:,:,:,2] = QVec[:,:,:,2]*gridSize - Qmax
        QVec[:,:,:,1] = QVec[:,:,:,1]*gridSize - Qmax 
        Qmag = np.sqrt((QVec[:,:,:,0])**2+(QVec[:,:,:,1])**2+(QVec[:,:,:,2])**2)
        dFF_xvw, SRO_xvw, scat_xvw = calculate_scattering(mol1,mol2,QVec,Qmag)

        # We want final shape to be 256*256*4, (so final final will be 256*256*12)
        # So permute and stack along z (where appropiate)
        scat_uxw = np.moveaxis(scat_uxw, 1, -1)
        dFF_uxw = np.moveaxis(dFF_uxw, 1, -1)
        SRO_uxw = np.moveaxis(SRO_uxw, 1, -1)

        scat_xvw = np.moveaxis(scat_xvw, 0, -1)
        dFF_xvw = np.moveaxis(dFF_xvw, 0, -1)
        SRO_xvw = np.moveaxis(SRO_xvw, 0, -1)

        # stack planes along second axis for saving
        dFF = np.concatenate((dFF_uxw,dFF_uvx,dFF_xvw), axis=2).astype(np.float32)
        SRO = np.concatenate((SRO_uxw,SRO_uvx,SRO_xvw), axis=2).astype(np.float32)
        scat = np.concatenate((scat_uxw,scat_uvx,scat_xvw), axis=2).astype(np.float32)    

        logger.info(">> Scattering complete in, saving arrays!")
        np.save(savefolder + molcode + "_" + str(conc_counter) + "_dFF.npy", dFF)
        np.save(savefolder + molcode + "_" + str(conc_counter) + "_SRO.npy", SRO)
        np.save(savefolder + molcode + "_" + str(conc_counter) + "_scat.npy", scat)

        # we want to save molcode, concentration, correlations
        # replicate 12 times in case some get deleted in next stage
        molcode_save = [molcode] * 12
        a_save = [a] * 12
        b_save = [b] * 12
        c_save = [c] * 12
        corr_save = [corr] * 12
        np.save(savefolder + molcode + "_" + str(conc_counter) + "_molcode_metadata.npy", np.asarray(molcode_save))
        np.save(savefolder + molcode + "_" + str(conc_counter) + "_a_metadata.npy", np.asarray(a_save))
        np.save(savefolder + molcode + "_" + str(conc_counter) + "_b_metadata.npy", np.asarray(b_save))
        np.save(savefolder + molcode + "_" + str(conc_counter) + "_c_metadata.npy", np.asarray(c_save))
        np.save(savefolder + molcode + "_" + str(conc_counter) + "_corr_metadata.npy", np.asarray(corr_save))

    t1 = time.time()
    logger.info(">> Molecules %s and %s complete in %.2f seconds, now assessing the Wasserstein similarity between images..."%(n1, n2, t1 - t0))
    Compile_Dataset.main(molcode=molcode, readfolder=savefolder, savefolder=imagefolder, artifactfolder="%s/../Artefacts/"%folder)

    logger.info(">> Completed post processing checks in %.2f seconds"%(time.time()-t1))
    # To save harddisk space, delete arrays we already have taken slices from

