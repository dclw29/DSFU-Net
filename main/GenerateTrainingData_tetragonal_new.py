import numpy as np
import matplotlib.pyplot as plt
from meta_library_2 import elements, numbers, atomic_SF, molecules
from itertools import combinations
import sys, os


sys.path.append("/data/lrudden/ML-DiffuseReader/main/")
import re
import Compile_Dataset


IAVectors = np.array([[1,0,0],[1,0,1],[2,0,0],[2,0,1],[2,0,2],[3,0,0],[3,0,1],[3,0,2],[3,0,3],[4,0,0],[4,0,1],
               [4,0,2],[4,0,3],[4,0,4],[5,0,0],[5,0,1],[5,0,2],[5,0,3],[5,0,4],[5,0,5]],dtype = float)


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
    counter = 0
    for j in range(len(IAVectors)): 
        if np.any(abs(IAVectors[j,:]) > 5): pass   
        else:
            vec = np.dot(rot, IAVectors[j,:])
            v1 = np.dot(rot,[-IAVectors[j,2],0,IAVectors[j,0]])
            v3 = np.dot(rot, -IAVectors[j,:])
            v4 = np.dot(rot,[IAVectors[j,2],0,-IAVectors[j,0]])
            SRO_cos = np.cos(np.einsum("ijkl,l->ijk", QVec, vec)) + np.cos(np.einsum("ijkl,l->ijk", QVec, v1)) + np.cos(np.einsum("ijkl,l->ijk", QVec, v3)) + np.cos(np.einsum("ijkl,l->ijk", QVec, v4))
            SRO += corr[j]*SRO_cos
            counter += 1

    SRO = SRO + 1            
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

folder = "/data/lrudden/ML-DiffuseReader/main"
input_group = [0,1,2,4]

TestSet = False
# Where will we save the data?
if TestSet:
    logger = logging.getLogger("log_%s_gen_data_test"%(str(input_group)))
    fh = logging.FileHandler("%s/logfiles/log_%s_validation.log"%(folder,str(input_group)))
    savefolder = "%s/../dataset/raw_files/validation/%s/"%(folder,str(input_group))
    imagefolder = "%s/../dataset/validation/%s/"%(folder,str(input_group))
else:
    #logger = logging.getLogger("log_%s_gen_data"%(str(input_group)))
    #fh = logging.FileHandler("%s/logfiles/log_%s.log"%(folder,str(input_group)))
    savefolder = "%s/../dataset/raw_files/%s/"%(folder,str(input_group))
    imagefolder = "%s/../dataset/training/%s/"%(folder,str(input_group))

if not os.path.exists(savefolder):
    os.mkdir(savefolder)
if not os.path.exists(imagefolder):
    os.mkdir(imagefolder)
for s in ["Scattering", "SRO", "dFF", "metadata"]:
    if not os.path.exists(imagefolder + s):
        os.mkdir(imagefolder + s)



f = open("metaTet.txt","w")
f.write("Molcode   Qmax   Concentration   Correlations \n")
# Pick two molecules from metadata
# loop through molecule groups (there are 5)
for g in [0,1,2,4]:
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
        for conc_counter in range(14): # 992 combinations in the dataset. ~12 different correlation values per molecule pair
            Qmax = np.random.rand()*2 + 6
            resolution = 256.0
            gridSize = (2*Qmax)/resolution  
           
            m1 = 0.5
            m2 = 1-m1
            amp = np.random.rand()*0.9+0.6
            decay = np.random.rand()+0.2
            freq = np.random.rand()+1
            x = np.arange(1,len(IAVectors)+1)
            corr = amp * np.exp(-decay*x) * np.cos(freq*x)

            s = ["{:.5f}".format(x) for x in corr]

            f.write(molcode+"  {:.3f}".format(Qmax)+"  {:.1f}".format(m1)+"  "+" ".join(s)+"\n")
            f.flush()

            # Re-calculate cell parameters based on concentrations
            # angles stay the same
            a = (mol1['lps'].values[0][0]*m2)+(mol2['lps'].values[0][0]*m1)
            b = (mol1['lps'].values[0][1]*m2)+(mol2['lps'].values[0][1]*m1)
            c = (mol1['lps'].values[0][2]*m2)+(mol2['lps'].values[0][2]*m1)


            alpha = 90.0
            gamma = 90.0
            beta = 90.0

            # Calculate reciprocal lattice
            rot = get_fractional_to_cartesian_matrix(a, b, c, alpha, beta, gamma,True)
            aVec = np.dot(rot,np.array([1,0,0]))
            bVec = np.dot(rot,np.array([0,1,0]))
            cVec = np.dot(rot,np.array([0,0,1]))
            astar = 2*np.pi*np.cross(bVec,cVec)/(np.dot(aVec,np.cross(bVec,cVec)))
            bstar = 2*np.pi*np.cross(cVec,aVec)/(np.dot(bVec,np.cross(cVec,aVec)))
            cstar = 2*np.pi*np.cross(aVec,bVec)/(np.dot(cVec,np.cross(aVec,bVec)))

            # Select hkl planes to calculate
            # calculate h0l, h1l and h2l planes
            # Create Q grid
            # set Qmax as 8 - just less than Qmax on BM0l at D = 200, lam = 0.7
            QVec,Qmag = make_q_grids([resolution,4,resolution],gridSize,Qmax,bstar[1])
            
            # calculate h0l, h1l, h2l and h3l planes
            dFF_uxw, SRO_uxw, scat_uxw = calculate_scattering(mol1,mol2,QVec,Qmag)
            ind = 0

            # this time we dont know where the scattering maxima will be so we can calculate a series of planes
            # between h0l and h1l and take the most intense
            #plt.imshow(scat_uxw.T,cmap = "hot", origin = "lower")
            #plt.colorbar()
            #plt.show()
            #plt.clf()

            # We want final shape to be 256*256*4
            # So permute and stack along z (where appropiate)
            scat = np.moveaxis(scat_uxw, 1, -1)
            dFF = np.moveaxis(dFF_uxw, 1, -1)
            SRO = np.moveaxis(SRO_uxw, 1, -1)
            #plt.clf()

            molcode_save = [molcode] * 12
            Qmax_save = [Qmax] * 12
            m1_save = [m1] * 12
            s_save = [s] * 12

            np.save(savefolder + molcode + "_" + str(conc_counter) + "_molcode_metadata.npy", np.asarray(molcode_save))
            np.save(savefolder + molcode + "_" + str(conc_counter) + "_qmax_metadata.npy", np.asarray(Qmax_save))
            np.save(savefolder + molcode + "_" + str(conc_counter) + "_m1_metadata.npy", np.asarray(m1_save))
            np.save(savefolder + molcode + "_" + str(conc_counter) + "_s_metadata.npy", np.asarray(s_save))

            np.save(savefolder + molcode+"_"+str(conc_counter)+"_dFF.npy", dFF)
            np.save(savefolder + molcode+"_"+str(conc_counter)+"_SRO.npy", SRO) 
            np.save(savefolder + molcode+"_"+str(conc_counter)+"_scat.npy", scat)   

        Compile_Dataset.main(molcode=molcode, readfolder=savefolder, savefolder=imagefolder, artifactfolder="%s/../Artefacts/"%folder)

f.close()


