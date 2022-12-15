import numpy as np
import random as rand
from _elements import elements_

def buildModel(BOX_SIZE,SITES,SITE_VARIANTS,VARIANT_ABUNDANCE):
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

    print("\n================Model construction================")
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

                    # Add disorder. If correlation coefficients are > -0.3 and < 0.3
                    #supercell[:,4] = 0
                    #seed = np.random.randint(len(supercell))
                    #count = 0
                    #supercell[seed,4] = 1
                    #while count < 0.5*len(supercell):
                    #    for i in range(len(neighbours[seed])):
                    #        if supercell[neighbours[seed,i],4] == 0:
                    #            supercell[neighbours[seed,i],4] = 1
                    #            count += 1
                    #    seed = np.random.choice(neighbours[seed])
#
                    #supercell[:,4] = (supercell[:,4]*2) -1

                    #if counter%2 == 0: 
                    #    supercell[address,4] = -1
                    #    defectCount[s] += 1
                    #counter += 1
                    """ Add more than two variants """
                    # add disorder
                    if SITE_VARIANTS[s]>1: 
                        if rand.random()<VARIANT_ABUNDANCE[s]:
                            supercell[address,4] = -1
                            defectCount[s] += 1
                    totalsites[s] += 1

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

