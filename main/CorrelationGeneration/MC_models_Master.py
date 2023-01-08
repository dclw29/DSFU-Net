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
from argparse import Namespace
import random 

##### IMPORT ALPHA CALCULATIONS
import SRO_sum

# Big loop MC function
def MC_LOOP(args, VARIANT_ABUNDANCE, TARGET_CORRELATIONS, FORCE_CONSTANTS):
    """
    Run the big MC loop to generate configurations to calculate alphas
    """

    # Build model
    modelType, rodAxis, stackingAxis, TARGET_CORRELATIONS = models.detOrderedModel(VARIANT_ABUNDANCE, TARGET_CORRELATIONS)
    supercell, VARIANT_ABUNDANCE = models.buildModel(args.BOX_SIZE, args.SITES, args.SITE_VARIANTS, VARIANT_ABUNDANCE, modelType, rodAxis, stackingAxis)

    # Calculate neighbours
    neighbours = models.getNeighbours(args.BOX_SIZE, args.SITES, args.SITE_VARIANTS, supercell, args.neighbourContacts, args.USE_PADDING)

    # Run MC
    supercell, correlations = Utils.MCOrdering(args.BOX_SIZE, args.SITE_ATOMS, VARIANT_ABUNDANCE, args.MC_CYCLES, args.MC_TEMP, TARGET_CORRELATIONS, FORCE_CONSTANTS, args.NEIGHBOUR_DIRECTIONS, supercell, neighbours, modelType, ncpus=args.ncpus)
    
    # reshape
    q = np.reshape(supercell[:,4], (BOX_SIZE[0],BOX_SIZE[1],BOX_SIZE[2]))
    q = np.where(q <0,0,q)

    # build buffer around q
    size_shift = len(q)
    # There is additional padding of 2 in args.SHIFT to account for running the checks from -I to I+1
    q = np.tile(q, (3,3,3))[size_shift - args.SHIFT[0]-2 : 2*size_shift+args.SHIFT[0]-2, size_shift - args.SHIFT[1]-2 : 2*size_shift+args.SHIFT[1]-2, size_shift - args.SHIFT[2]-2 : 2*size_shift+args.SHIFT[2]-2]

    # calculate alphas
    # prep generator
    TOTAL_NUM = SRO_sum.mass_calc_3d(q, IJK=args.IJK, shift=args.SHIFT, ma=VARIANT_ABUNDANCE[0])
    alpha = []
    counter = 0
    for item in TOTAL_NUM:
        #print(item) -  contains information on I J K vectors and sum
        alpha.append(item[-1])
        counter += 1
        if counter % 200 == 0:
            print(item)

    return VARIANT_ABUNDANCE, alpha

##### INITALISING ####
rand.seed()
filestem = sys.argv[1]
run = int(sys.argv[2])

# Read input file
# Some parameters will change later on
CELL,BOX_SIZE,USE_PADDING,SITES,SITE_ATOMS,SITE_VARIANTS,VARIANT_ABUNDANCE,TARGET_CORRELATIONS,FORCE_CONSTANTS,NEIGHBOUR_DIRECTIONS,MC_CYCLES,MC_TEMP,neighbourContacts,atoms = imports.readParamFile(f'{filestem}.params')

# Create IJK list
IJK = SRO_sum.pre_gen_ijk(7, 7, 7)
np.save("IJK_vectors.npy", np.asarray(IJK))
# Add padding to account for running checks between -I and I+1
SHIFT = [7+2, 7+2, 7+2]

# Create args list
args = Namespace(CELL=CELL, BOX_SIZE=BOX_SIZE, USE_PADDING=USE_PADDING, SITES=SITES, SITE_ATOMS=SITE_ATOMS, SITE_VARIANTS=SITE_VARIANTS, NEIGHBOUR_DIRECTIONS=NEIGHBOUR_DIRECTIONS,
                 MC_CYCLES=MC_CYCLES, MC_TEMP=MC_TEMP, neighbourContacts=neighbourContacts, atoms=atoms, IJK=IJK, SHIFT=SHIFT, ncpus=4)

# Run MC loop
# variant abundance will be random between 0.1 and 0.5 inclusive
# target correlations in principle can be between -0.6 and 0.6 but not under certain conditions relative to variant abundance, alter target correlation if this condition is met
# specifically, calculate alpha min based on variant abundance and this sets the min threahshold for target correlations
# force constants are based on the values of the correlations

# Run 250 times for all molecular combinations
# Generate 4 alphas per concentation, doing 3 concentrations (variant abundance)
t0 = time.time()
alpha_list = []
abundance_target_list = []
for n in range(250):
    for C in range(3): # concentration / variant abundance
        VARIANT_ABUNDANCE[0] = (random.random() * 0.4) + 0.1
        minAlpha = max(- VARIANT_ABUNDANCE[0] / (1 - VARIANT_ABUNDANCE[0]), -0.6)
        for a in range(4): # gen 4 
            # create along each dimension
            X = (random.random() * (0.6+abs(minAlpha))) + minAlpha
            Y = (random.random() * (0.6+abs(minAlpha))) + minAlpha
            Z = (random.random() * (0.6+abs(minAlpha))) + minAlpha

            while abs(X) < 0.05 and abs(Y) < 0.05 and abs(Z) < 0.05:
                print(">> X Y and Z for alpha gen stage all very close to zero! Resetting...")
                X = (random.random() * (0.6+abs(minAlpha))) + minAlpha
                Y = (random.random() * (0.6+abs(minAlpha))) + minAlpha
                Z = (random.random() * (0.6+abs(minAlpha))) + minAlpha                
            
            TARGET_CORRELATIONS = [X, Y, Z]

            for cnt, val in enumerate(TARGET_CORRELATIONS):
                if val < -0.05: FORCE_CONSTANTS[cnt] = 100
                elif val > 0.05: FORCE_CONSTANTS[cnt] = -100
                else: FORCE_CONSTANTS[cnt] = 0

            print(VARIANT_ABUNDANCE[0], TARGET_CORRELATIONS, a)

            VARIANT_ABUNDANCE, alpha = MC_LOOP(args, VARIANT_ABUNDANCE, TARGET_CORRELATIONS, FORCE_CONSTANTS)
            alpha_list.append(alpha)
            abundance_target_list.append(VARIANT_ABUNDANCE + TARGET_CORRELATIONS) #  last three numbers in each row are therefore the target correlations

    np.save("Alphas.npy", np.asarray(alpha_list))
    np.save("Abundance_Target_list.npy", np.asarray(abundance_target_list))
    print(">> Run %i took %.2f seconds"%(int(n), time.time() - t0))

