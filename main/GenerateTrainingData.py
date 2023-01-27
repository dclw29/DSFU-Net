import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from meta_library_2 import elements, numbers, atomic_SF, molecules
from itertools import combinations, permutations
import sys
import math

correlations = np.load("alphas_test.npy")
concs = np.load("abundance_list_test.npy")
counter = 0
IAVectors = np.load("Vectors.npy")

Qmax = 8.0
resolution = 256.0
gridSize = (2*Qmax)/resolution

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
        if np.any(abs(IAVectors[i,:]) > 2): pass   
        else:
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

'''
TO DISCUSS:
useful print statements?
Only use 5 neighbours? - need to remove vectors containing 6,-6,7,-7 from IJK and the corresponding alpha values - need to find the indexes for [1,0,0],[0,1,0] and [0,0,1] if you do this
filenames - leading zeros
'''

# Pick two molecules from metadata
# loop through molecule groups (there are 5)
for g in range(5):
    # get the list of molecule names
    names = list(molecules.query('group == @g')['name'])

    # get the unique combinations of molecules
    a = list(combinations(names,2))
    molComb = np.asarray([list(x) for x in list(a)])

    # extract info for the pair
    for pair in range(len(molComb)): #len(molComb)
        n1 = molComb[pair,0]
        n2 = molComb[pair,1]
        molcode = str(n1)+str(n2)
        
        print("Doing molecules "+n1+" and "+n2)
        idx1 = molecules[molecules['name']==str(n1)].index
        idx2 = molecules[molecules['name']==str(n2)].index
        mol1 = molecules.iloc[idx1]        
        mol2 = molecules.iloc[idx2]

        # loop over multiple correlations 
        for c in range(12): # 992 combinations in the dataset. ~12 different correlation values per molecule pair
            m1 = concs[counter,0]
            m2 = 1-m1
            corr = correlations[counter,:]
            counter += 1
            print(m1)

            # Re-calculate cell parameters based on concentrations
            # angles stay the same
            a = (mol1['lps'].values[0][0]*m2)+(mol2['lps'].values[0][0]*m1)
            b = (mol1['lps'].values[0][1]*m2)+(mol2['lps'].values[0][1]*m1)
            c = (mol1['lps'].values[0][2]*m2)+(mol2['lps'].values[0][2]*m1)

            print(a,b,c)
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

            # Get main correlations (number correct for 7 shells)
            cx = corr[225]
            cy = corr[15]
            cz = corr[1]

            print(cx,cy,cz)

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
                plt.plot(summed)
                plt.show()
                plt.clf()

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
            highest_scatter_x = peaks_x[np.argpartition(heights_x, -len(peaks_x))[-len(peaks_x):]]-10+int(resolution/2)
            highest_scatter_z = peaks_z[np.argpartition(heights_z, -len(peaks_z))[-len(peaks_z):]]-10+int(resolution/2)

            if len(peaks_x) < 4: 
                plane_x = np.random.randint(int(resolution/2),high=int(resolution),size = 4-len(peaks_x))
                highest_scatter_x = np.append(highest_scatter_x,plane_x)
            if len(peaks_z) < 4: 
                plane_z = np.random.randint(int(resolution/2),high=int(resolution),size = 4-len(peaks_x))
                highest_scatter_z = np.append(highest_scatter_z,plane_z)

            # Find the corresponding QVec
            q_uvx = highest_scatter_x*gridSize - Qmax
            q_xvw = highest_scatter_z*gridSize - Qmax

            # calculate new scattering planes
            QVec,Qmag = make_q_grids([resolution,resolution,4],gridSize,Qmax,cstar[2])
            dFF_uvx, SRO_uvx, scat_uvx = calculate_scattering(mol1,mol2,QVec,Qmag)

            QVec,Qmag = make_q_grids([4,resolution,resolution],gridSize,Qmax,astar[0])
            dFF_xvw, SRO_xvw, scat_xvw = calculate_scattering(mol1,mol2,QVec,Qmag)

            # reshape the arrays so they can be stacked, final shape will be 256*12*256
            scat_uvx = np.moveaxis(scat_uvx, -1, 1)
            dFF_uvx = np.moveaxis(dFF_uvx, -1, 1)
            SRO_uvx = np.moveaxis(SRO_uvx, -1, 1)

            scat_xvw = np.moveaxis(scat_xvw, 0, 1)
            dFF_xvw = np.moveaxis(dFF_xvw, 0, 1)
            SRO_xvw = np.moveaxis(SRO_xvw, 0, 1)

            # stack planes along second axis for saving
            dFF = np.hstack((dFF_uxw,dFF_uvx,dFF_xvw))
            SRO = np.hstack((SRO_uxw,SRO_uvx,SRO_xvw))
            scat = np.hstack((scat_uxw,scat_uvx,scat_xvw))    

            #np.save(molcode+"_"+str(c)+"_dFF.npy", dFF)
            #np.save(molcode+"_"+str(c)+"_SRO.npy", SRO) 
            #np.save(molcode+"_"+str(c)+"_scat.npy", scat)  


# Plot some graphs
for i in range(1):
    

    plt.imshow(dFF[:,i,:].T,cmap = "hot", origin = "lower")
    plt.colorbar()
    plt.show()
    plt.clf()

    plt.imshow(SRO[:,i,:].T,cmap = "hot", origin = "lower")
    plt.colorbar()
    plt.show()
    plt.clf()

    plt.imshow(scat[:,i,:].T,cmap = "hot", origin = "lower")
    plt.colorbar()
    plt.show()
    plt.clf()



