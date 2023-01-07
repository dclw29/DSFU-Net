import numpy as np
import random as rand
from _elements import elements_

def detOrderedModel(VARIANT_ABUNDANCE, TARGET_CORRELATIONS):

    print("\n================Model construction================\n")
    minAlpha = -min(VARIANT_ABUNDANCE)/(1-min(VARIANT_ABUNDANCE))
    TARGET_CORRELATIONS = np.asarray(TARGET_CORRELATIONS)
    if min(TARGET_CORRELATIONS)< minAlpha:
        print("\n WARNING: The desired correlations are too low for \n          this concentration of defects.")
        key_pressed = input(' Press 0 to set the target correlations to their minimum value\n Press 1 to set the target correlations to (requested correlation * the minimum)\n')
        if key_pressed == '0':
            TARGET_CORRELATIONS = np.where(TARGET_CORRELATIONS<minAlpha,minAlpha,TARGET_CORRELATIONS)
            print(" The new target correlations are: \n    "+str(TARGET_CORRELATIONS))
        elif key_pressed == '1':
            TARGET_CORRELATIONS = np.where(TARGET_CORRELATIONS<minAlpha,TARGET_CORRELATIONS*abs(minAlpha),TARGET_CORRELATIONS)
            print(" The new target correlations are: \n    "+str(TARGET_CORRELATIONS))

    modelType = 0
    rodAxis = -1
    StackingAxis = -1
    
    lpos = np.greater(TARGET_CORRELATIONS,0.4)
    lneg = np.less(TARGET_CORRELATIONS,0.4*minAlpha)
    small = np.logical_and(np.greater_equal(TARGET_CORRELATIONS,0.4*minAlpha), np.less_equal(TARGET_CORRELATIONS,0.4))

    directions = {0:"x",1:"y",2:"z"}

    #print(lpos,lneg,small)
    #print(np.sum(lneg))
    # 3D Random
    if np.sum(small) == 3:
        modelType = 0
        print(" Starting from model type " + str(modelType) + ":\n   3D random defects")
    # 3D Clustered
    elif np.sum(lpos) == 3:
        modelType = 1
        print(" Starting from model type " + str(modelType) + ":\n   3D clustered defects")
    # 3D Ordered
    elif np.sum(lneg) == 3:
        modelType = 2
        print(" Starting from model type " + str(modelType) + ":\n   3D ordered defects")
    # 2D Ordered rods, randomly arranged
    elif np.sum(lneg) == 1 and np.sum(small) == 2:
        modelType = 3
        rodAxis = int(np.where(lneg == True)[0])
        print(" Starting from model type " + str(modelType) + ":\n   Ordered rods along "+ directions[rodAxis])
    # 2D Clustered rods, randomly arranged
    elif np.sum(lpos) == 1 and np.sum(small) == 2:
        modelType = 4
        rodAxis = int(np.where(lpos == True)[0])
        print(" Starting from model type " + str(modelType) + ":\n   Clustered rods along "+ directions[rodAxis])
    # 1D Ordered layers, randomly stacked
    elif np.sum(lneg) == 2 and np.sum(small) == 1:
        modelType = 5
        StackingAxis = int(np.where(small == True)[0])
        print(" Starting from model type " + str(modelType) + ":\n   Ordered layers, randomly stacked along "+ directions[StackingAxis])
    # 1D Clustered layers, randomly stacked
    elif np.sum(lpos) == 2 and np.sum(small) == 1:
        modelType = 6
        StackingAxis = int(np.where(small == True)[0])
        print(" Starting from model type " + str(modelType) + ":\n   Clustered layers, randomly stacked along "+ directions[StackingAxis])
    # 1D layers ordered in one clustered in other, randomly stacked
    elif np.sum(lneg) == 1 and np.sum(lpos) == 1 and np.sum(small) == 1:
        modelType = 7
        StackingAxis = int(np.where(small == True)[0])
        rodAxis = int(np.where(lpos == True)[0])
        print(" Starting from model type " + str(modelType) + ":\n   Clustered rods along "+ directions[rodAxis]+", ordered within layers\n   and randomly stacked along "+ directions[StackingAxis])
    # 1D Ordered layers, stacked in clusters
    elif np.sum(lneg) == 2 and np.sum(lpos) == 1:
        modelType = 8
        StackingAxis = int(np.where(lpos == True)[0])
        print(" Starting from model type " + str(modelType) + ":\n   Clustered rods along " + directions[StackingAxis] + ", ordered normal to rods")
    # 1D Clustered layers stacked in order
    elif np.sum(lpos) == 2 and np.sum(lneg) == 1:
        modelType = 9
        StackingAxis = int(np.where(lneg == True)[0])
        print(" Starting from model type " + str(modelType) + ":\n   Clustered layers stacked in order along " + directions[StackingAxis])

    return modelType, rodAxis, StackingAxis, list(TARGET_CORRELATIONS)

def buildModel(BOX_SIZE,SITES,SITE_VARIANTS,VARIANT_ABUNDANCE,modelType,rodAxis,stackingAxis):
    '''
    Creates the supercell model for Monte Carlo ordering
        The supercell model is constructed from unit cells.
        Each unit cell contains a number of sites. 
        The sites could represent individual atoms, or molecules, groups of atoms, layers etc.
        Sites are stored in a numpy array called supercell, with dimensions (number of sites * 5).
        Each site has 5 bits of information ascociated with it:
            supercell[site,0] - which unit cell the site is in along the x direction
            supercell[site,1] - which unit cell the site is in along the y direction
            supercell[site,2] - which unit cell the site is in along the z direction
            supercell[site,3] - the number of the site within each unit cell
            supercell[site,4] - a value of 1 or -1 to denote whether this site is and ideal (1) or defect (-1) site
        The sites can be accessed using their index (or address) which is equal to (x*(ny)*(nz)+(y*(nz))+z)*SITES + s
    
    Parameters:
        BOX_SIZE (np.ndarray(3,dtype=int)): The size of the supercell in x,y,z in terms of unitcells
        SITES (int): The number of sites in each unit cell
        SITE_VARIANTS (np.ndarray(SITES,dtype = int)): The number of variants that can occupy each site
        VARIANT_ABUNDANCE (np.ndarray(SITES,dtype = float)): The abundances/occupancies of those variants 
    Returns: 
        supercell (np.ndarray((number of sites * 5),dtype = int)): model containing site information
        VARIANT_ABUNDANCE (np.ndarray(SITES,dtype = float): contains updated variant abundances consistent with the constructed model
    '''
    print("\n   Number of sites: "+str(SITES))

    nx = BOX_SIZE[0]
    ny = BOX_SIZE[1]
    nz = BOX_SIZE[2]
    ncells = nx*ny*nz
    supercell = np.full((SITES*ncells,5),1)
    rand.seed()
    defectCount = np.zeros(SITES)
    totalsites = np.zeros(SITES)

    counter = 0
    for x in range(nx): # loop over unit cells along x
        for y in range(ny): # loop over unit cells along y
            for z in range(nz): # loop over unit cells along z
                for s in range(SITES): # loop over sites in a unit cell
                    address = (x*(ny)*(nz)+(y*(nz))+z)*SITES + s
                    supercell[address,0] = x
                    supercell[address,1] = y
                    supercell[address,2] = z
                    supercell[address,3] = s
                    #if supercell[address,1] == 0: print(address) 

    if modelType == 0:
        for i in range(len(supercell)):
            if SITE_VARIANTS[supercell[i,3]]>1: 
                if rand.random()<VARIANT_ABUNDANCE[supercell[i,3]]:
                    supercell[i,4] = -1
                    defectCount[supercell[i,3]] += 1
            totalsites[supercell[i,3]] += 1
        
    elif modelType == 1:
        seed = np.random.randint(len(supercell))
        count = 0
        stop = False
        supercell[seed,4] = -1
        cells = int((VARIANT_ABUNDANCE[supercell[seed,3]]*len(supercell))**(1/3)) # divided by number of sites
        for i in range(cells+1):
            if stop: break
            else:
                for j in range(cells+1):
                    if stop: break
                    else:
                        for k in range(cells+1):
                            if stop: break
                            else:
                                for l in range(SITES):
                                    if supercell[seed,0]+i > nx-1: x = supercell[seed,0]+i-nx
                                    else: x = i+supercell[seed,0]
                                    if supercell[seed,1]+j > ny-1: y = supercell[seed,1]+j-ny
                                    else: y = j+supercell[seed,1]
                                    if supercell[seed,2]+k > nz-1: z = supercell[seed,2]+k-nz
                                    else: z = k+supercell[seed,2]
                                    s = supercell[seed,3] + l
                                    address = (x*(ny)*(nz)+(y*(nz))+z)*SITES + s
                                    supercell[address,4] = -1
                                    count += 1
                                    defectCount[supercell[seed,3]] += 1
                                    if count > VARIANT_ABUNDANCE[supercell[i,3]]*len(supercell): stop = True
        # using neighbours - maybe useful for middle correlations 
        #while count < VARIANT_ABUNDANCE[supercell[i,3]]*len(supercell):
        #    for i in range(len(neighbours[seed])):
        #        if supercell[neighbours[seed,i],4] == 1:
        #            supercell[neighbours[seed,i],4] = -1
        #            defectCount[supercell[neighbours[seed,i],3]] += 1
        #            count += 1
        #    seed = np.random.choice(neighbours[seed])
        totalsites[:] = nx*ny*nz
        
    elif modelType == 2:
        order = np.full((BOX_SIZE[0],BOX_SIZE[1],BOX_SIZE[2]),1)
        for t in range(int(1/VARIANT_ABUNDANCE[0])):
            if rand.random()<VARIANT_ABUNDANCE[0]: 
                order[0,0,t] = -1
                break
        for u in range(t,BOX_SIZE[2],int(1/VARIANT_ABUNDANCE[0])):
            order[0,0,u] = -1
        for j in range(1,BOX_SIZE[1]):
            order[0,j,:] = np.roll(order[0,j-1,:],1)
        for k in range(1,BOX_SIZE[0]):
            order[k,:,:] = np.roll(order[k-1,:,:],1,axis=1)

        for k in range(len(supercell)):
            supercell[k,4] = order[supercell[k,0],supercell[k,1],supercell[k,2]] 
            if order[supercell[k,0],supercell[k,1],supercell[k,2]] == -1: counter += 1
        defectCount[0] = counter
        totalsites[:] = nx*ny*nz

    elif modelType == 3:
        axes = np.array([0,1,2])
        axes = np.delete(axes,rodAxis)
        order = np.full((BOX_SIZE[0],BOX_SIZE[1],BOX_SIZE[2]),1)
        if rodAxis == 0:
            for i in range(np.shape(order)[1]):
                for j in range(np.shape(order)[2]):
                    for t in range(int(1/VARIANT_ABUNDANCE[0])):
                        if rand.random()<VARIANT_ABUNDANCE[0]: 
                            order[t,i,j] = -1
                            break
                    for u in range(t,BOX_SIZE[0],int(1/VARIANT_ABUNDANCE[0])):
                        order[u,i,j] = -1
        elif rodAxis == 1:
            for i in range(np.shape(order)[0]):
                for j in range(np.shape(order)[2]):
                    for t in range(int(1/VARIANT_ABUNDANCE[0])):
                        if rand.random()<VARIANT_ABUNDANCE[0]: 
                            order[i,t,j] = -1
                            break
                    for u in range(t,BOX_SIZE[1],int(1/VARIANT_ABUNDANCE[0])):
                        order[i,u,j] = -1
        elif rodAxis == 2:
            for i in range(np.shape(order)[0]):
                for j in range(np.shape(order)[1]):
                    for t in range(int(1/VARIANT_ABUNDANCE[0])):
                        if rand.random()<VARIANT_ABUNDANCE[0]: 
                            order[i,j,t] = -1
                            break
                    for u in range(t,BOX_SIZE[2],int(1/VARIANT_ABUNDANCE[0])):
                        order[i,j,u] = -1


        for k in range(len(supercell)):
            supercell[k,4] = order[supercell[k,0],supercell[k,1],supercell[k,2]] 
            if order[supercell[k,0],supercell[k,1],supercell[k,2]] == -1: counter += 1
        defectCount[0] = counter
        totalsites[:] = nx*ny*nz 
        
    elif modelType == 4:
        axes = np.array([0,1,2])
        axes = np.delete(axes,rodAxis)
        order = np.full((BOX_SIZE[axes[0]],BOX_SIZE[axes[1]]),1)
        for i in range(np.shape(order)[0]):
            for j in range(np.shape(order)[1]):
                if rand.random()<VARIANT_ABUNDANCE[0]: order[i,j] = -1
        
        for k in range(len(supercell)):
            supercell[k,4] = order[supercell[k,axes[0]],supercell[k,axes[1]]] 
            if order[supercell[k,axes[0]],supercell[k,axes[1]]] == -1: counter += 1
        defectCount[0] = counter
        totalsites[:] = nx*ny*nz  

    elif modelType == 5:
        order = np.full((BOX_SIZE[0],BOX_SIZE[1],BOX_SIZE[2]),1)
        
        if stackingAxis == 0:
            for i in range(BOX_SIZE[stackingAxis]):
                for t in range(int(1/VARIANT_ABUNDANCE[0])):
                    if rand.random()<VARIANT_ABUNDANCE[0]: 
                        order[i,0,t] = -1
                        break
                for u in range(t,BOX_SIZE[2],int(1/VARIANT_ABUNDANCE[0])):
                    order[i,0,u] = -1
                for j in range(1,BOX_SIZE[1]):
                    order[i,j,:] = np.roll(order[i,j-1,:],1)
        elif stackingAxis == 1:
            for i in range(BOX_SIZE[stackingAxis]):
                for t in range(int(1/VARIANT_ABUNDANCE[0])):
                    if rand.random()<VARIANT_ABUNDANCE[0]: 
                        order[0,i,t] = -1
                        break
                for u in range(t,BOX_SIZE[2],int(1/VARIANT_ABUNDANCE[0])):
                    order[0,i,u] = -1
                for j in range(1,BOX_SIZE[1]):
                    order[j,i,:] = np.roll(order[j-1,i,:],1)
        elif stackingAxis == 2:
            for i in range(BOX_SIZE[stackingAxis]):
                for t in range(int(1/VARIANT_ABUNDANCE[0])):
                    if rand.random()<VARIANT_ABUNDANCE[0]: 
                        order[0,t,i] = -1
                        break
                for u in range(t,BOX_SIZE[2],int(1/VARIANT_ABUNDANCE[0])):
                    order[0,u,i] = -1
                for j in range(1,BOX_SIZE[1]):
                    order[j,:,i] = np.roll(order[j-1,:,i],1)

        for k in range(len(supercell)):
            supercell[k,4] = order[supercell[k,0],supercell[k,1],supercell[k,2]] 
            if order[supercell[k,0],supercell[k,1],supercell[k,2]] == -1: counter += 1
        defectCount[0] = counter
        totalsites[:] = nx*ny*nz

    elif modelType == 6:
        counter = 0
        # 1D ordered layers
        order = np.full(BOX_SIZE[stackingAxis],1)
        for i in range(len(order)):
            if rand.random()<VARIANT_ABUNDANCE[0]: 
                order[i] = -1

        for k in range(len(supercell)):
            supercell[k,4] = order[supercell[k,stackingAxis]] 
            if order[supercell[k,stackingAxis]] == -1: counter += 1
        defectCount[0] = counter
        totalsites[:] = nx*ny*nz    

    elif modelType == 7:
        order = np.full((BOX_SIZE[0],BOX_SIZE[1],BOX_SIZE[2]),1)
        orderedAxis = np.delete([0,1,2],max(stackingAxis,rodAxis))
        orderedAxis = int(np.delete(orderedAxis,min(stackingAxis,rodAxis)))

        if stackingAxis == 0 and orderedAxis == 1 and rodAxis == 2:
            for i in range(BOX_SIZE[stackingAxis]):
                for t in range(int(1/VARIANT_ABUNDANCE[0])):
                    if rand.random()<VARIANT_ABUNDANCE[0]: 
                        order[i,t,:] = -1
                        break
                for u in range(t,BOX_SIZE[orderedAxis],int(1/VARIANT_ABUNDANCE[0])):
                        order[i,u,:] = -1
        elif stackingAxis == 0 and orderedAxis == 2 and rodAxis == 1:
            for i in range(BOX_SIZE[stackingAxis]):
                for t in range(int(1/VARIANT_ABUNDANCE[0])):
                    if rand.random()<VARIANT_ABUNDANCE[0]: 
                        order[i,:,t] = -1
                        break
                for u in range(t,BOX_SIZE[orderedAxis],int(1/VARIANT_ABUNDANCE[0])):
                        order[i,:,u] = -1
        elif stackingAxis == 1 and orderedAxis == 0 and rodAxis == 2:
            for i in range(BOX_SIZE[stackingAxis]):
                for t in range(int(1/VARIANT_ABUNDANCE[0])):
                    if rand.random()<VARIANT_ABUNDANCE[0]: 
                        order[t,i,:] = -1
                        break
                for u in range(t,BOX_SIZE[orderedAxis],int(1/VARIANT_ABUNDANCE[0])):
                        order[u,i,:] = -1
        elif stackingAxis == 1 and orderedAxis == 2 and rodAxis == 0:
            for i in range(BOX_SIZE[stackingAxis]):
                for t in range(int(1/VARIANT_ABUNDANCE[0])):
                    if rand.random()<VARIANT_ABUNDANCE[0]: 
                        order[:,i,t] = -1
                        break
                for u in range(t,BOX_SIZE[orderedAxis],int(1/VARIANT_ABUNDANCE[0])):
                        order[:,i,u] = -1
        elif stackingAxis == 2 and orderedAxis == 0 and rodAxis == 1:
            for i in range(BOX_SIZE[stackingAxis]):
                for t in range(int(1/VARIANT_ABUNDANCE[0])):
                    if rand.random()<VARIANT_ABUNDANCE[0]: 
                        order[t,:,i] = -1
                        break
                for u in range(t,BOX_SIZE[orderedAxis],int(1/VARIANT_ABUNDANCE[0])):
                        order[u,:,i] = -1
        elif stackingAxis == 2 and orderedAxis == 1 and rodAxis == 0:
            for i in range(BOX_SIZE[stackingAxis]):
                for t in range(int(1/VARIANT_ABUNDANCE[0])):
                    if rand.random()<VARIANT_ABUNDANCE[0]: 
                        order[:,t,i] = -1
                        break
                for u in range(t,BOX_SIZE[orderedAxis],int(1/VARIANT_ABUNDANCE[0])):
                        order[:,u,i] = -1

        for k in range(len(supercell)):
            supercell[k,4] = order[supercell[k,0],supercell[k,1],supercell[k,2]] 
            if order[supercell[k,0],supercell[k,1],supercell[k,2]] == -1: counter += 1
        defectCount[0] = counter
        totalsites[:] = nx*ny*nz 

    elif modelType == 8:
        order = np.full((BOX_SIZE[0],BOX_SIZE[1],BOX_SIZE[2]),1)
        
        if stackingAxis == 0:
            for t in range(int(1/VARIANT_ABUNDANCE[0])):
                if rand.random()<VARIANT_ABUNDANCE[0]: 
                    order[:,0,t] = -1
                    break
            for u in range(t,BOX_SIZE[2],int(1/VARIANT_ABUNDANCE[0])):
                order[:,0,u] = -1
            for j in range(1,BOX_SIZE[1]):
                order[:,j,:] = np.roll(order[:,j-1,:],1)
        elif stackingAxis == 1:
            for t in range(int(1/VARIANT_ABUNDANCE[0])):
                if rand.random()<VARIANT_ABUNDANCE[0]: 
                    order[0,:,t] = -1
                    break
            for u in range(t,BOX_SIZE[2],int(1/VARIANT_ABUNDANCE[0])):
                order[0,:,u] = -1
            for j in range(1,BOX_SIZE[1]):
                order[j,:,:] = np.roll(order[j-1,:,:],1)
        elif stackingAxis == 2:
            for t in range(int(1/VARIANT_ABUNDANCE[0])):
                if rand.random()<VARIANT_ABUNDANCE[0]: 
                    order[t,0,:] = -1
                    break
            for u in range(t,BOX_SIZE[0],int(1/VARIANT_ABUNDANCE[0])):
                order[u,0,:] = -1
            for j in range(1,BOX_SIZE[1]):
                order[:,j,:] = np.roll(order[:,j-1,:],1,axis=0)

        for k in range(len(supercell)):
            supercell[k,4] = order[supercell[k,0],supercell[k,1],supercell[k,2]] 
            if order[supercell[k,0],supercell[k,1],supercell[k,2]] == -1: counter += 1
        defectCount[0] = counter
        totalsites[:] = nx*ny*nz

    elif modelType == 9:
        counter = 0
        # 1D ordered layers
        order = np.full(BOX_SIZE[stackingAxis],1)
        for i in range(int(1/VARIANT_ABUNDANCE[0])):
            if rand.random()<VARIANT_ABUNDANCE[0]: 
                order[i] = -1
                break
        for j in range(i,len(order),int(1/VARIANT_ABUNDANCE[0])):
                order[j] = -1

        for k in range(len(supercell)):
            supercell[k,4] = order[supercell[k,stackingAxis]] 
            if order[supercell[k,stackingAxis]] == -1: counter += 1
        defectCount[0] = counter
        totalsites[:] = nx*ny*nz
        
    # update the variant abundances with the actual value from the constructed model
    for s in range(SITES):
        VARIANT_ABUNDANCE[s] = (defectCount[s]/totalsites[s])
        if defectCount[s] != 0: print("   "+str(int(defectCount[s]))+" defects added on site "+str(s+1)+": {:.1f}%".format((defectCount[s]/totalsites[s])*100))

    return supercell, VARIANT_ABUNDANCE

def getNeighbours(BOX_SIZE,SITES,SITE_VARIANTS, supercell, neighbours, USE_PADDING):
    '''
    Finds the neighbours of each site in the model and stores the indices in a list.
        Reads the contact vectors from neighbours and applies them to each of the sites in the model.
        The position of each neighbour (i.e. its index in the supercell array) are appended to a list 
        and the completed list is stored in its own array.
    Parameters:
        BOX_SIZE (np.ndarray(3,dtype=int)): The size of the supercell in x,y,z in terms of unitcells
        SITES (int): The number of sites in each unit cell
        SITE_VARIANTS (np.ndarray(SITES,dtype = int)): The number of variants that can occupy each site
        supercell (np.ndarray((number of sites * 5),dtype = int)): model containing site information
        neighbours (dictionary): contains the contact vectors for each neighbour for each site, as read in from the input file
        USE_PADDING (bool): if you want to exclude the surface of the supercell from the Monte Carlo (e.g. if there are layers normal to 111 making the usual periodic boundary definition of neighbours void)
                        you can add 1 unit cell's width padding.
    Returns: 
        neighbourArray (np.ndarray((number of sites * max neighbours),dtype = int)):
            For each site in the supercell, neighbourArray stores a list of the indices of the sites neighbours
    '''
    nx = BOX_SIZE[0]
    ny = BOX_SIZE[1]
    nz = BOX_SIZE[2]
    ncells = nx*ny*nz

    # Find the maximum number of neighbours
    nneighbours = []
    for i in range(SITES):
        contacts = neighbours[f'S{i+1}_NC']
        nneighbours.append(len(contacts))
    maxNeighbours = np.max(nneighbours)

    # Find neighbours for each site based on contact vectors given in input file
    neighbourArray = np.full((ncells*SITES,maxNeighbours),-1)
    for a in range(len(supercell)):
        x = supercell[a,0]
        y = supercell[a,1]
        z = supercell[a,2]
        s = supercell[a,3]

        if SITE_VARIANTS[s]>1: 
            contacts = neighbours[f'S{s+1}_NC']
            for n in range(maxNeighbours):
                contact = contacts[f'n{n+1}']
                if x+contact[0] < 0: x1 = x+contact[0]+nx
                elif x+contact[0] >= nx: x1 = x+contact[0]-nx
                else: x1 = x+contact[0]
                if y+contact[1] < 0: y1 = y+contact[1]+ny
                elif y+contact[1] >= ny: y1 = y+contact[1]-ny
                else: y1 = y+contact[1]
                if z+contact[2] < 0: z1 = z+contact[2]+nz
                elif z+contact[2] >= nz: z1 = z+contact[2]-nz
                else: z1 = z+contact[2]
                s1 = s+contact[3]

                neighbourAddress = (x1*(ny)*(nz)+(y1*(nz))+z1)*SITES + s1
                neighbourArray[a,n] = neighbourAddress
    return neighbourArray

def convertToAtoms(BOX_SIZE,SITE_ATOMS,supercell,atoms):
    '''
    Converts the supercell to an atomic model.
        Depending on the spin of each site from supercell, convert this to an array of atoms ready for export to files.  
            The atomicModel array contains 1 row for each atom and 5 columns:
                atomicModel[atom,0] - the proton number of the specific element on that site
                atomicModel[atom,1] - the proton number of the element most commonly found on that site (useful if sites are partially occupied)
                atomicModel[atom,2:4] - the x,y,z coordinates of the atoms, given as fractional coordinates of its UNIT CELL (not the whole supercell)
    Parameters:
        BOX_SIZE (np.ndarray(3,dtype=int)): The size of the supercell in x,y,z in terms of unitcells
        SITE_ATOMS (np.ndarray(SITES,dtype = int)): The number of atoms located on each site
        supercell (np.ndarray((number of sites,5),dtype = int)): model containing site information
        atoms (dictionary): contains the elements and coordinates for each site as read from the input file.
    Returns: 
        atomicModel (np.ndarray((number of atoms,5),dtype = float)): model containing all atoms, their elements and coordinates
    '''
    natoms = sum(SITE_ATOMS)
    nx = BOX_SIZE[0]
    ny = BOX_SIZE[1]
    nz = BOX_SIZE[2]
    ncells = nx*ny*nz
    atomsArray = np.zeros((natoms*ncells, 5))


    for a in range(len(supercell)):
        s = supercell[a,3]
        if supercell[a,4] == 1: n=1
        else: n=2
        site = atoms[f'S{s+1}_V{n}'] 
        for o in range(SITE_ATOMS[s]):
            atomsArray[(a*SITE_ATOMS[s])+o,0] = elements_[site[f'a{o+1}'][0]]
            atomsArray[(a*SITE_ATOMS[s])+o,1] = elements_[site[f'a{o+1}'][1]]
            atomsArray[(a*SITE_ATOMS[s])+o,2:] = site[f'a{o+1}'][2:]

    return atomsArray