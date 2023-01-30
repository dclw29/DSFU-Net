"""
We have numerous files generated for the alpha generation (to maximise speed)
COmbine and clean them
Note we want no abundances > 0.55
"""

import numpy as np
import os, sys

# Parameters
num=11 # number of files to loop through

abundance = np.load("Abundance_Target_list_0.npy")
alpha = np.load("Alphas_0.npy")
for n in range(1, num):
    abundance = np.concatenate((abundance, np.load("Abundance_Target_list_%i.npy"%n)))
    alpha = np.concatenate((alpha, np.load("Alphas_%i.npy"%n)))

IJK = np.load("IJK_vectors_1.npy") # all IJK arrays are the same

concentration = abundance[:, 0]
targets = abundance[:, 1:]

# remove all concentrations above 0.55
avoid = concentration > 0.55

concentration = concentration[~avoid]
alpha = alpha[~avoid]
targets = targets[~avoid]

# save
np.save("./FINAL/concentrations.npy", concentration)
np.save("./FINAL/alphas.npy", alpha)
np.save("./FINAL/target_correlations.npy", targets)
np.save("./FINAL/IJK_vectors.npy", IJK)