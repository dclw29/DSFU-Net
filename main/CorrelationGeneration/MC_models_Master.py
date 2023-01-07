##############################################################################
#   Diffuse scattering analysis code
#   Chloe Fuller, 2022
##############################################################################

import MC_models_Building as models
import MC_models_Utils_parr as Utils
import MC_models_Exports as exports
import sys
import time
import MC_models_Imports as imports
import numpy as np
import random as rand
import matplotlib.pyplot as plt
rand.seed()
filestem = sys.argv[1]
run = int(sys.argv[2])

def V_true(C, i, i_shift, j, j_shift,k,k_shift):
    """
    Return true of the vector i, j corresponds to positive 1 (i.e. a sequence of A B), sequence of B A would return minus 1 (so take abs as B A also counts for A B to be adjacent)
    i, j current location in matrix, i_shift and j_shift are indiced of V
    """
    x = np.shape(C)[0]
    c = np.tile(C,(3,3,3))

    if c[i+x, j+x,k+x] - c[i+x+i_shift, j+x+j_shift,k+x+k_shift] == 1:
        return True
    else:
        return False 

def V_sum(C, i, j,k,ma):
    """
    How many times does a sequence A, B occur over the vector direction {i, j} in the crystal C?
    Note we always want to use a sliding window for the crystal so we can maximise information (i.e. don't skip potential off center sequences)
    If i or j are negative, do we need to start counting from the other side of the crystal?
    """
    num = 0
    # width of array possibility
    W = np.shape(C)[0]
    # and height
    H = np.shape(C)[1]

    D = np.shape(C)[2]
    counter = 0
    for w in range(W):
        for h in range(H):
            for d in range(D):
                num += V_true(C, w, i, h, j,d,k)
                counter += 1
    prob = num/counter
    al = 1-(prob/(ma*(1-ma)))

    return al # number of occurances for V indices i and j

def mass_calc(C, VARIANT_ABUNDANCE, I=1,J=1,K=1):
    """
    From i=0 to i=I and j=0 to j=J (indices of matrix), calculate the number of occurances of the sequence 
    Generators list of current indices i and j and the number of occurances
    """
    ma = np.average(VARIANT_ABUNDANCE)
    for i in range(0,I+1):
        for j in range(-J,J+1):
            for k in range(-K,K+1):
                if i == 0 and j == 0 and k < 0: continue
                elif i == 0 and j < 0: continue 
                else: yield ["%i %i %i"%(i,j,k), V_sum(C, i, j,k,ma)]


def initialise(N):
    """
    Create crystal of random 1s and 0s (for testing), 1 = mol A, 0 = mol B
    """
    #return np.random.randint(2, size=(N,N))
    x = np.full((N,N),-1,dtype=int)
    x[1::2,::2] = 1
    x[::2,1::2] = 1
    return x

CELL,BOX_SIZE,USE_PADDING,SITES,SITE_ATOMS,SITE_VARIANTS,VARIANT_ABUNDANCE,TARGET_CORRELATIONS,FORCE_CONSTANTS,NEIGHBOUR_DIRECTIONS,MC_CYCLES,MC_TEMP,neighbourContacts,atoms = imports.readParamFile(f'{filestem}.params')

modelType, rodAxis, stackingAxis, TARGET_CORRELATIONS = models.detOrderedModel(VARIANT_ABUNDANCE, TARGET_CORRELATIONS)

supercell, VARIANT_ABUNDANCE = models.buildModel(BOX_SIZE,SITES,SITE_VARIANTS,VARIANT_ABUNDANCE,modelType,rodAxis,stackingAxis)
#q = np.reshape(supercell[:,4], (BOX_SIZE[0],BOX_SIZE[1],BOX_SIZE[2]))
#q = np.where(q <0,0,q) + 1
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#x,y,z = q.nonzero()
#ax.set_xlabel("x")
#ax.set_ylabel("y")
#ax.set_zlabel("z")
#ax.scatter(x, y, z, c=q, alpha=0.5,cmap = 'bwr')
#plt.show()

neighbours = models.getNeighbours(BOX_SIZE,SITES,SITE_VARIANTS, supercell, neighbourContacts, USE_PADDING)
#t2 = time.time()
t0 = time.time()
supercell, correlations = Utils.MCOrdering(BOX_SIZE,SITE_ATOMS,VARIANT_ABUNDANCE,MC_CYCLES,MC_TEMP, TARGET_CORRELATIONS, FORCE_CONSTANTS, NEIGHBOUR_DIRECTIONS,supercell,neighbours,modelType,ncpus=4)
t1 = time.time()
print("time: {:.4f}s".format(t1-t0))


#plt.imshow(q[:,:,int(BOX_SIZE[2]/2)])
#plt.show()
#print(q)

#atomArray = models.convertToAtoms(BOX_SIZE,SITE_ATOMS,supercell,atoms)
#average,Uiso = Utils.averageUnitCell(CELL,BOX_SIZE,SITE_ATOMS,atomArray,USE_PADDING)
#exports.Scatty(filestem,run,atomArray,average,CELL,BOX_SIZE,SITE_ATOMS,USE_PADDING)


TOTAL_NUM = mass_calc(q)
for item in TOTAL_NUM:
    print(item)
