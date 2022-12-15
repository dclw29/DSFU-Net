import numpy as np
import time
from collections import Counter
from numba import jit
import numba as nb
from math import ceil


def calcCorrelation(VARIANT_ABUNDANCE,supercell,neighbours,nd,direction):
    
    ma = np.average(VARIANT_ABUNDANCE)
    mb = 1-ma
    neighbourList = neighbours[:,direction*nd[direction]:direction*nd[direction]+nd[direction]]
    count = 0
    energy = 0
    for n in range(np.shape(neighbourList)[1]):
        e = (supercell[:,4]+1)/2 - (supercell[neighbourList[:,n],4]+1)/2
        count += len(e)   
        energy += np.sum(np.where(e == 1, e, 0))
        prob = float(energy/count)
    correlation = 1-(prob/(ma*mb))
    '''
    if mode == "Layers":
        axis = {0:modelParams.nx, 1:modelParams.ny, 2:modelParams.nz}
        stackingAxis = siteParams.stackingAxis
        nlayers = siteParams.nsites
        variantPercentages = siteParams.variantPercentages
        ma = np.mean(variantPercentages)/100.
        mb = 1-ma

        #energy = 0
        #count = 0
        #for i in range(axis[stackingAxis]):
        #    address = i*nlayers + site
        #    neighbourList = getNeighbourList(neighbours,site,address,direction,neighboursInDirections)
        #    count += len(neighbourList)
        #    energy += calcIsingEnergy(supercell,address,neighbourList,1)
    #
        #correlation = float(energy / count)

        #using warren-Cowley correlation parameters : 1-(p/mamb)
        count = 0.
        
        for a in range(axis[stackingAxis]):
            for s in site:
                address = a*nlayers + s
                neighbourList = getNeighbourList(neighbours,s,address,direction,neighboursInDirections)
                if supercell[address] + supercell[neighbourList[0]] == -2:
                    count += 1

        prob = ma - (count / (float(len(supercell))))
        correlation = 1 - (prob/(ma*mb))
        #print(ma, count,prob, len(supercell)/nlayers, ma*mb, prob/(ma*mb))
    '''
    return correlation

def calcIsingEnergy(supercell, neighbourList, k):

    energy = np.zeros(np.shape(supercell))
    for n in range(np.shape(neighbourList)[2]):
        energy = energy + supercell*neighbourList[:,:,n]*k

    return energy

def checkConvergence(lst,criteria):
    # Function to check convergence of correlations
    # Stops MC cycles if it gets stuck
    ele = lst[0]
    chk = True
    for item in lst:
        if abs(ele - item) > criteria:
            chk = False
            break
    return chk 

def averageUnitCell(CELL,BOX_SIZE,SITE_ATOMS,atomsArray,USE_PADDING):
    #calculate average positions 
    if USE_PADDING: padding = 1
    else: padding = 0
    natoms = sum(SITE_ATOMS)
    nx = BOX_SIZE[0]
    ny = BOX_SIZE[1]
    nz = BOX_SIZE[2]
    ncells = (nx-2*padding)*(ny-2*padding)*(nz-2*padding)
    a = CELL[0]
    b = CELL[1]
    c = CELL[2]

    Uiso = np.zeros(natoms)
    average = np.zeros((natoms,8))     
    for l in range(natoms):
        tota = 0.
        totx = 0.
        toty = 0.
        totz = 0.
        siteatoms = []
        for i in range(padding,nx-padding):
            for j in range(padding,ny-padding):
                for k in range(padding,nz-padding):
                    address = ((i*(ny)*(nz))+(j*(nz))+k)*natoms + l
                    if atomsArray[address,1] != 0: 
                        tota += 1
                        siteatoms.append(atomsArray[address,1])
                    totx += atomsArray[address,2]
                    toty += atomsArray[address,3]
                    totz += atomsArray[address,4]
        t = Counter(siteatoms)

        average[l,0] = float(tota/ncells) # overall occupancy
        average[l,1] = float(totx/ncells) # x
        average[l,2] = float(toty/ncells) # y
        average[l,3] = float(totz/ncells) # z
        for o in range(len(list(t.keys()))):
            average[l,4+(2*o)] = int(list(t.keys())[o])    # element 1
            average[l,5+(2*o)] = t[list(t.keys())[o]]/tota    # percentage of element 1
    for i in range(natoms):
        dx2 = 0.
        dy2 = 0.
        dz2 = 0.
        counter = 0
        for x in range(padding,nx-padding):
            for y in range(padding,ny-padding):
                for z in range(padding,nz-padding):
                    address = ((x*ny*nz)+(y*nz)+z)*natoms + i
                    if atomsArray[address,1] != 0:  
                        dx = (atomsArray[address,2] - average[i,1])*a
                        if dx > 0.5*a: dx = dx-a
                        if dx < -0.5*a: dx = dx+a
                        dx2 += dx*dx
                        dy = (atomsArray[address,3] - average[i,2])*b
                        if dy > 0.5*b: dy = dy-b
                        if dy < -0.5*b: dy = dy+b
                        dy2 += dy*dy
                        dz = (atomsArray[address,4] - average[i,3])*c
                        if dz > 0.5*c: dz = dz-c
                        if dz < -0.5*c: dz = dz+c
                        dz2 += dz*dz
                        counter += 1
        if counter == 0:
            sqdx2 = 0
            sqdy2 = 0
            sqdz2 = 0
        else:
            sqdx2 = np.sqrt(dx2/counter)
            sqdy2 = np.sqrt(dy2/counter)
            sqdz2 = np.sqrt(dz2/counter)
        Uiso[i] = ((sqdx2 + sqdy2 + sqdz2)/3)
    
    return average, Uiso

def updateForces(target, current, k):
    # changes the force constants for specified sites and directions to drive 
    # the system towards the target correlations
    newk = np.zeros(np.shape(k))
    assert np.shape(target) == np.shape(k)
    assert np.shape(current) == np.shape(k)
    for d in range(len(target)):
        # If current correlation == target, the force constant with be divided by 1
        #divisor = (0.5+(current[d]/(current[d]+target[d]+0.00001))) #+0.001 ish to prevent /0 errors

        #newk[d] = (k[d]/divisor)
        newk[d] = k[d] + (current[d] - target[d])*100
        if newk[d] < -1000000: newk[d] = -1000000
        elif newk[d] > 1000000: newk[d] = 1000000
    return newk

def MCOrdering(BOX_SIZE,SITE_ATOMS,VARIANT_ABUNDANCE,DIMENSIONS,MC_CYCLES,MC_TEMP, TARGET_CORRELATIONS, FORCE_CONSTANTS, NEIGHBOUR_DIRECTIONS,supercell,neighbours):
    # Calculate the initial correlations for all specified sites and contacts
    # sites is a list of the sites (or layers) you want correlations to be calculated for
    # Note that all the sites in the list will be allowed to swap spins with each other
    # If you want to order multiple sites separately, you need to run the MCOrdering multiple times.
    # correlations is a list of the desired correlations for each site and each site-neighbour interaction
    # Its a sites*max_neighbours array with the element value giving the desired correlation
    # If you want to ignore some neighbour contacts, set the desired correlation to 10

    targetCorrelations = np.asarray(TARGET_CORRELATIONS)
    k = np.asarray(FORCE_CONSTANTS)
    neighboursInDirections = np.asarray(NEIGHBOUR_DIRECTIONS)

    natoms = sum(SITE_ATOMS)
    ncells = BOX_SIZE[0]*BOX_SIZE[1]*BOX_SIZE[2]
    totsites = len(supercell)

    convCrit = 0.05
    convList = [0]*10
    sm = (ceil(BOX_SIZE[0]/10))*(ceil(BOX_SIZE[1]/10))*(ceil(BOX_SIZE[2]/10))
    bufferSize = int((1/np.amin(VARIANT_ABUNDANCE))*100000)
    move = np.zeros((sm,2),dtype=int)
    spins = np.zeros((sm,2),dtype=int)
    selected = np.zeros(len(supercell),dtype=int)

    print("===================MC Ordering====================")
    print("\n   Initial correlations: ")
    
    for direction in range(len(targetCorrelations)):
        if targetCorrelations[direction] != 10.: 
            init = calcCorrelation(VARIANT_ABUNDANCE,supercell,neighbours,neighboursInDirections,direction)
            print("       Correlation "+str(direction+1)+": "+str(round(init,3)))

    # Monte Carlo swapping algorithm.
    # Randomly pick two sites in the supercell, swap their pseudo-spins compare energies before and after
    # and accept/reject the move depending on the Metropolis condition 
    nmoves = int((natoms*ncells)/sm)

    t1 = 0
    t2 = 0
    t3 = 0
    t01 = 0
    t02 = 0
    t03 = 0
    rng = np.random.default_rng()
   
    pbest = 10.
    best = 10.
    stop = False
    
    for c in range(MC_CYCLES):
        if not stop:
            if c % 5 == 0: print("\n          -----------Cycle " + str(c)+"----------")
            accept = 0
            reject = 0
            if best < pbest: pbest = best
            #run x in parallel, select loswest energy config. concurrent
            """
            if multithread and num != 1:
            import concurrent.futures

            # setup parameters
                param_list = []
                splits = math.ceil(num / threadpool)
                for idx in range(threadpool):
                    if idx == 0:
                        num0 = (idx * splits) + 2 # ranges for append
                    else:
                        num0 = (idx * splits)
                    if idx == threadpool-1:
                        num1 = num+1
                    else:
                        num1 = (idx * splits) + splits

                    param_list.append([data, num0, num1, filename, coupled])

                # now run with multithreading
                # This cool module is able to manage worker load without join() or wait() statements
                with concurrent.futures.ThreadPoolExecutor(max_workers=threadpool) as executor:
                    future = [executor.submit(_inner_load_, param[0], param[1], param[2], param[3], param[4]) for param in param_list]

                # data stored in future[idx].result(), need to create large datastructure
                # remove first file set (12 seq) as was used as handle
                for f in future:
                    data = np.concatenate((data, np.asarray(f.result()[12:])), axis=0)
                seq_size = len(data[0][:-1])
            """
            for m in range(nmoves):
                t_ = time.time()
                move[:] = move[:]*0
                spins[:] = spins[:]*0
                selected[:] = selected[:]*0  

                rnum_buffer = np.random.randint(0,totsites,size = bufferSize)  
                ri = 0

                for mo in range(len(move)):
                
                    t = time.time()
                    #randomly pick first site 
                    address1 = rnum_buffer[ri]
                    ri += 1
                    while selected[address1] == 1:
                        address1 = rnum_buffer[ri]
                        ri += 1
                    spin1 = supercell[address1,4]
                    spin2 = supercell[address1,4]

                    t01 += time.time()-t
                    t = time.time()

                    #find different site 2
                    address2 = rnum_buffer[ri]
                    ri += 1
                    while spin2 == spin1 or selected[address2] == 1:
                        address2 = rnum_buffer[ri]
                        ri += 1
                        spin2 = supercell[address2,4]

                    t02 += time.time()-t
                    t = time.time()

                    # record selected addresses and neighbours so that they are not picked again           
                    selected[address1] = 1
                    selected[neighbours[address1]] = 1
                    selected[address2] = 1
                    selected[neighbours[address2]] = 1

                    t03 += time.time()-t
                    t = time.time()

                    # store moves
                    move[mo,0] = address1
                    move[mo,1] = address2
                    spins[mo,0] = supercell[address1,4]
                    spins[mo,1] = supercell[address2,4]



                t1 += time.time()-t_
                t = time.time()

                #Calculate energies
                energy = np.zeros(np.shape(move))
                for d in range(len(targetCorrelations)):

                    neighbourList = neighbours[move,d*neighboursInDirections[d]:d*neighboursInDirections[d]+neighboursInDirections[d]]
                    energy += calcIsingEnergy(supercell[move,4], supercell[neighbourList,4], k[d])

                EtotOld = energy.sum(1)

                #Swap layers
                supercell[move[:,0],4] = spins[:,1]
                supercell[move[:,1],4] = spins[:,0]

                ##Calculate new energies
                energyNew = np.zeros(np.shape(move))
                for d in range(len(targetCorrelations)):

                    neighbourList = neighbours[move,d*neighboursInDirections[d]:d*neighboursInDirections[d]+neighboursInDirections[d]]
                    energyNew += calcIsingEnergy(supercell[move,4], supercell[neighbourList,4], k[d])

                EtotNew = energyNew.sum(1)
                t2 += time.time()-t
                t = time.time()

                #Accept with probability
                deltaE = EtotNew-EtotOld
                rnums = rng.random(len(move))
                metCond = np.exp(-(deltaE)/MC_TEMP)-0.5
                outcome = np.less_equal(rnums,metCond)

                accept += np.sum(outcome)
                reject += len(outcome)-np.sum(outcome)

                for i in range(len(move)):
                    if outcome[i] == False:
                        #reject += 1
                        supercell[move[i,0],4] = spins[i,0]
                        supercell[move[i,1],4] = spins[i,1]

                t3 += time.time()-t
                t = time.time() 
            # Calculates ratio of accepted:rejected moves during cycle
            # For the most efficient sampling, we want it to be 0.5
            # To achieve this we alter the 'temperature' of the MC simulation
            # When the ratio is ideal (0.5) temp is divided by 1
            accept_ratio = accept/(accept +  reject)
            MC_TEMP = MC_TEMP/(0.5 + accept_ratio)
            if MC_TEMP > 10000: MC_TEMP = 10000
            #print("accept ratio: {:.4f}".format(accept_ratio))
            #print("Temperature: {:.4f}".format(MC_TEMP))

            # Calculate the correlations in all specified directions
            currentCorrelations = np.full(np.shape(targetCorrelations),-10.)
            if c==MC_CYCLES-1: print("\n       ----------Final correlations----------")
            for direction in range(len(targetCorrelations)):
                currentCorrelations[direction] = calcCorrelation(VARIANT_ABUNDANCE,supercell,neighbours,neighboursInDirections,direction)
                if (c>0 and c % 5 == 0) or c==MC_CYCLES-1: print("       Current correlation "+str(direction+1)+": "+str(round(float(currentCorrelations[direction]),3))+
                                         "  Target: "+str(float(targetCorrelations[direction])))

            #Do some checks at the end of every cycle
            corrDiff = abs(targetCorrelations - currentCorrelations)
            best = np.sum(corrDiff)
            
            # If the target is reached, stop cycling and output the current config
            if np.all(corrDiff < 0.025):
                stop = True
                copy = np.copy(supercell)
                print("\n--------------------------------------------------")
                print("                Targets reached!")
                print("--------------------------------------------------")
            
            # If this config improves on the previous best one, make a note
            else:
                if best < pbest:
                    copy = np.copy(supercell)

                # Check correlation values for convergence, stops the algorithm running if it gets stuck 
                convList.append(currentCorrelations[0])
                del convList[0]
                if checkConvergence(convList,convCrit): 
                    # If convergence is reached by the current correlations are not the target value, shuffle a small part of the array
                    # to avoid getting stuck in local minima 
                    if np.any(abs(targetCorrelations - currentCorrelations) > 0.1):
                        start = np.random.randint(0, len(supercell)-len(supercell)/20)
                        np.random.shuffle(supercell[start:start+int(len(supercell)/20)])
                        print("shuffled")
                    else:
                        print("\n--------------------------------------------------")
                        print("               Convergence reached!")
                        print("--------------------------------------------------")
                        stop = True
                
                    
        # keep forces constant for the first 5 cycles
        # Otherwise update the force constants to drive the correlations towards the desired ones.
        
        #if(c < 5): continue
        #else: 
        k = updateForces(targetCorrelations, currentCorrelations, k)

    for direction in range(len(targetCorrelations)):
        currentCorrelations[direction] = calcCorrelation(VARIANT_ABUNDANCE,copy,neighbours,neighboursInDirections,direction)
        print("       Best Correlation "+str(direction+1)+": "+str(round(currentCorrelations[direction],3))+
                                         "  Target: "+str(float(targetCorrelations[direction])))
        
    print("\n==================================================")
    #print("Pick sites: {:.4f}s".format(t1))
    #print("01: {:.4f}s".format(t01))
    #print("02: {:.4f}s".format(t02))
    #print("03: {:.4f}s".format(t03))
    #print("Energies: {:.4f}s".format(t2))
    #print("accept moves: {:.4f}s".format(t3))
    return copy, currentCorrelations