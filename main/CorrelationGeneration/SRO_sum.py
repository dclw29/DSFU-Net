"""
Count number of times sequence A,B occurs over various crystal directions
Extend to 3D, number of times A, B and C occurs over various crystal direction
"""

import numpy as np
from time import time as t

def initialise_2d(N):
    """
    Create crystal of random 1s and 0s (for testing), 1 = mol A, 0 = mol B
    """
    return np.random.randint(2, size=(N,N))

def V_true_2d(C, i, i_shift, j, j_shift):
    """
    Return true of the vector i, j corresponds to positive 1 (i.e. a sequence of A B), sequence of B A would return minus 1 (so take abs as B A also counts for A B to be adjacent)
    i, j current location in matrix, i_shift and j_shift are indiced of V
    """
    #if np.abs(C[i, j] - C[i+i_shift, j+j_shift]) == 1:
    # not testing B to A
    if C[i, j] - C[i+i_shift, j+j_shift] == 1:
        return True
    else:
        return False 

def V_sum_2d(C, i, j, ma=0.5):
    """
    How many times does a sequence A, B occur over the vector direction {i, j} in the crystal C?
    Note we always want to use a sliding window for the crystal so we can maximise information (i.e. don't skip potential off center sequences)
    If i or j are negative, do we need to start counting from the other side of the crystal?
    ma is a parameter
    """
    num = 0
    # width of array possibility
    W = np.shape(C)[0]
    # and height
    H = np.shape(C)[1]
    counter = 0
    for w in range(W):
        for h in range(H):
            num += V_true_2d(C, w, i, h, j)
            counter += 1
    prob = num/counter
    alpha = 1 - (prob/(ma*(1-ma)))
    return alpha

def mass_calc_2d(C, I=5, J=5, ma=0.5):
    """
    From i=0 to i=I and j=0 to j=J (indices of matrix), calculate the number of occurances of the sequence 
    Generators list of current indices i and j and the number of occurances
    """
    for i in range(I):
        for j in range(J):
            yield ["%i %i"%(i,j), V_sum_2d(C, i, j, ma)]

def tile_2d(C, I, J):
    """
    Return a new array, C, which is size C + I, J along each direction (with the border swapped over)
    Essentially, minimal peiodic boundary conditions
    """

    # create placeholder array
    P = np.zeros((C.shape[0]+2*I, C.shape[1]+2*J))
    P[I:-I, J:-J] += C

    # append before start
    P[:I, J:-J] += C[-I:, :]
    P[I:-I, :J] += C[:, -J:]
    P[:I, :J] += C[-I:, -J:] # top left corner
    P[-I:, :J] += C[:I, -J:] # bottom left corner

    # second half
    P[-I:, J:-J] += C[:I, :]
    P[I:-I, -J:] += C[:, :J]
    P[:I, -J:] += C[-I:, :J] # top right corner
    P[-I:, -J:] += C[:I, :J] # bottom right corner    
    
    return P.astype(int)

def main_2d(init=100):
    """
    Test the 2D code
    """
    t0 = t()
    C = initialise_2d(init)
    C = tile_2d(C, abs(I)+additional_padding, abs(J)+additional_padding) 
    TOTAL_NUM = mass_calc_2d(C)
    for item in TOTAL_NUM:
        print(item)
    t1 = t()
    print("Total time: %.2f seconds"%(t1-t0))

def initialise_3d(N):
    """
    Create crystal of random 1s and 0s (for testing), 1 = mol A, 0 = mol B
    """
    return np.random.randint(2, size=(N,N,N))

def initialise_3d_checkerboard(N):
    """
    Create crystal of checkerboard pattern of 0s and 1s
    """
    x = np.zeros((N,N), dtype=bool)
    x[1::2,::2] = 1
    x[::2,1::2] = 1
    xN = ~x
    X = np.concatenate((x[:,:,None], xN[:,:,None]), axis=2).astype(int)

    return np.tile(X, int(N/2))

def V_true_3d(C, i, i_shift, j, j_shift, k, k_shift, shift):
    """
    Return true of the vector i, j corresponds to positive 1 (i.e. a sequence of A B), sequence of B A would return minus 1 (so take abs as B A also counts for A B to be adjacent)
    i, j current location in matrix, i_shift and j_shift are indiced of V
    Extended to 3D with K
    """
    # avoid the first few values (given by shift)
    if C[i+shift[0], j+shift[1], k+shift[2]] - C[i+i_shift+shift[0], j+j_shift+shift[1], k+k_shift+shift[2]] == 1:
        return True
    else:
        return False 

def V_sum_3d(C, i, j, k, shift, ma=0.5):
    """
    How many times does a sequence A, B occur over the vector direction {i, j} in the crystal C?
    Note we always want to use a sliding window for the crystal so we can maximise information (i.e. don't skip potential off center sequences)
    If i or j are negative, do we need to start counting from the other side of the crystal?
    ma is a parameter
    """
    num = 0
    # width of array possibility
    W = np.shape(C)[0] - 2*shift[0] # ignore initial region
    # and height
    H = np.shape(C)[1] - 2*shift[1]
    # Depth
    D = np.shape(C)[2] - 2*shift[2]
    counter = 0
    for w in range(W):
        for h in range(H):
            for d in range(D):
                num += V_true_3d(C, w, i, h, j, d, k, shift)
                counter += 1
    prob = num/counter
    alpha = 1 - (prob/(ma*(1-ma)))
    return alpha

#def mass_calc_3d(C, I=5, J=5, K=5, ma=0.5):
#    """
#    From i=0 to i=I and j=0 to j=J (indices of matrix), calculate the number of occurances of the sequence 
#    Generators list of current indices i and j and the number of occurances
#    Parse also the shift parameters to start counting the from (for boundary conditions)
#    """
#    for i in range(0, I+1):
#        for j in range(-J, J+1):
#            for k in range(-K, K+1):
#                if i == 0 and j == 0 and k < 0: continue
#                elif i == 0 and j < 0: continue
#                else: yield ["%i %i %i"%(i,j,k), V_sum_3d(C, i, j, k, [abs(I), abs(J), abs(K)], ma)]


def mass_calc_3d(C, IJK, shift, ma=0.5):
    """
    From i=0 to i=I and j=0 to j=J (indices of matrix), calculate the number of occurances of the sequence 
    Generators list of current indices i and j and the number of occurances
    Parse also the shift parameters to start counting the from (for boundary conditions)
    """
    for l in IJK:
        yield ["%i %i %i"%(l[0], l[1], l[2]), V_sum_3d(C, l[0], l[1], l[2], shift, ma)]

def tile_3d(C, I, J, K):
    """
    Return a new array, C, which is size C + I, J, K along each direction (with the border swapped over)
    Essentially, minimal peiodic boundary conditions
    """

    # create placeholder array
    P = np.zeros((C.shape[0]+2*I, C.shape[1]+2*J, C.shape[2]+2*K))
    P[I:-I, J:-J, K:-K] += C

    # append before start
    P[:I, J:-J, K:-K] += C[-I:, :, :]
    P[I:-I, :J, K:-K] += C[:, -J:, :]
    P[I:-I, J:-J, :K] += C[:, :, -K:]

    # second half
    P[-I:, J:-J, K:-K] += C[:I, :, :]
    P[I:-I, -J:, K:-K] += C[:, :J, :] 
    P[I:-I, J:-J, -K:] += C[:, :, :K]  
    
    # corners
    P[:I, :J, :K] += C[-I:, -J:, -K:] 
    P[-I:, :J, :K] += C[:I, -J:, -K:]
    P[-I:, -J:, :K] += C[:I, :J, -K:]
    P[-I:, -J:, -K:] += C[:I, :J, :K]
    
    P[-I:, :J, -K:] += C[:I, -J:, :K]
    P[:I, :J, -K:] += C[-I:, -J:, :K]
    P[:I, -J:, :K] += C[-I:, :J, -K:]
    P[:I, -J:, -K:] += C[-I:, :J, :K]

    # need edges also....

    return P.astype(int)

def pre_gen_ijk(I, J, K):
    """
    Pre-generate the I J K lists so we don't need giant ifelse statements in generator
    """
    ijk = []
    for i in range(0, I+1):
        for j in range(-J, J+1):
            for k in range(-K, K+1):
                if i == 0 and j == 0 and k < 0: continue
                elif i == 0 and j < 0: continue
                else: ijk.append([i, j, k])
    return ijk

def main_3d(init=50, I=5, J=5, K=5, additional_padding=2):
    """
    Test the 2D code
    """
    t0 = t()
    #C = initialise_3d(init)
    C = initialise_3d_checkerboard(init)
    # Need to tile to account for border, but don't need excess tiling
    #C = tile_3d(C, abs(I)+additional_padding, abs(J)+additional_padding, abs(K)+additional_padding) 
    C = np.tile(C, (3,3,3))[init-I:2*init+I, init-J:2*init+J, init-K:2*init+K]
    IJK = pre_gen_ijk(I, J, K)
    TOTAL_NUM = mass_calc_3d(C, IJK, [I+additional_padding, J+additional_padding, K+additional_padding])
    for item in TOTAL_NUM:
        print(item)
    t1 = t()
    print("Total time: %.2f seconds"%(t1-t0))