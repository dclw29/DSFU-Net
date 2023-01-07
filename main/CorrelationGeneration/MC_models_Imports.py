import sys

def checkKey(dic):
    key = ["CELL","BOX_SIZE","SITES","SITE_ATOMS","SITE_VARIANTS","VARIANT_ABUNDANCE","TARGET_CORRELATIONS","FORCE_CONSTANTS","NEIGHBOUR_DIRECTIONS","MC_CYCLES","MC_TEMP"]
    for k in key:
        if k not in list(dic.keys()):
            print(" Parameter file is missing "+k)
            sys.exit()

def readParamFile(filename):
    print("\n================Reading from input================\n")
    file1 = open(filename)
    print(" Filename: "+filename)
    # Read first section of file to get cell parameters etc
    args = {}
    for line in file1:
        if not line.strip().startswith("#"):
            if "NEIGHBOUR CONTACTS" in line: break
            else:
                line = line.strip()
                values = line.split("=")
                if len(values) >= 2:
                    args[values[0].strip()] = values[1].strip()
    # check that all the necessary parameters were given 
    checkKey(args)

    # convert to usable parameters
    CELL = [float(i) for i in args["CELL"].strip().split()]
    print(" Cell parameters: "+str(CELL[0])+", "+str(CELL[1])+", "+str(CELL[1]))
    print("                  "+str(CELL[3])+", "+str(CELL[4])+", "+str(CELL[5]))
    if len(CELL) != 6: 
        print("Wrong number of cell parameters")
        sys.exit()

    BOX_SIZE = [int(i) for i in args["BOX_SIZE"].strip().split()]
    print(" Box size: "+str(BOX_SIZE[0])+", "+str(BOX_SIZE[1])+", "+str(BOX_SIZE[2]))
    if len(BOX_SIZE) != 3: 
        print("Wrong number of supercell parameters")
        sys.exit()

    if "USE_PADDING" in args.keys():
        USE_PADDING = args["USE_PADDING"].strip()
    else:
        USE_PADDING = False

    SITES = int(args["SITES"].strip())

    SITE_ATOMS = [int(i) for i in args["SITE_ATOMS"].strip().split()]
    if len(SITE_ATOMS) != SITES:
        if len(SITE_ATOMS) == 1:
            SITE_ATOMS = SITE_ATOMS * SITES
        else: 
            print("The number of site atoms does not match the number of sites.")
            sys.exit()
    
    SITE_VARIANTS = [int(i) for i in args["SITE_VARIANTS"].strip().split()]
    if len(SITE_VARIANTS) != SITES:
        if len(SITE_VARIANTS) == 1:
            SITE_VARIANTS = SITE_VARIANTS * SITES
        else: 
            print("The number of site variants does not match the number of sites.")
            sys.exit()


    VARIANT_ABUNDANCE = [float(i) for i in args["VARIANT_ABUNDANCE"].strip().split()]
    if len(VARIANT_ABUNDANCE) != SITES:
        if len(VARIANT_ABUNDANCE) == 1:
            VARIANT_ABUNDANCE = VARIANT_ABUNDANCE * SITES
        else: 
            print("The number of site variants does not match the number of sites.")
            sys.exit()

    TARGET_CORRELATIONS = [float(i) for i in args["TARGET_CORRELATIONS"].strip().split()]
    FORCE_CONSTANTS = [float(i) for i in args["FORCE_CONSTANTS"].strip().split()]
    NEIGHBOUR_DIRECTIONS = [int(i) for i in args["NEIGHBOUR_DIRECTIONS"].strip().split()]
    assert len(TARGET_CORRELATIONS) == len(FORCE_CONSTANTS), "The number of target correlations and force constants does not match"
    assert len(TARGET_CORRELATIONS) == len(NEIGHBOUR_DIRECTIONS), "The number of target correlations and neighbours does not match"
    
    MC_CYCLES = int(args["MC_CYCLES"].strip())
    MC_TEMP = float(args["MC_TEMP"].strip())

    neighbourContacts = {}
    stop = False
    for i in range(1, SITES+1):
        if stop: break
        a = {}
        for line in file1:
            if "ATOMS" in line: 
                stop = True
                break
            elif f"SITE{i+1}" in line: break
            else:
                line = line.strip()
                values = line.split("=")
                if len(values) >= 2:
                    a[values[0].strip()] = [int(i) for i in values[1].strip().split()]
            neighbourContacts[f'S{i}_NC'] = a

    atoms = {}
    stop = False
    for i in range(1, SITES+1):
        for j in range(1,SITE_VARIANTS[i-1]+1):
            if stop: break
            a = {}
            for line in file1:
                if line == "\n": 
                    stop = True
                    break
                elif f"SITE{i+1}" in line: break
                elif f"VARIANT{j+1}" in line: break
                else:
                    line = line.strip()
                    values = line.split("=")
                    if len(values) >= 2:
                        a[values[0].strip()] =  list(values[1].strip().split())
                        a[values[0].strip()][2:] = [float(i) for i in  a[values[0].strip()][2:]]
            atoms[f'S{i}_V{j}'] = a

    file1.close()
    return(CELL,BOX_SIZE,USE_PADDING,SITES,SITE_ATOMS,SITE_VARIANTS,VARIANT_ABUNDANCE,TARGET_CORRELATIONS,FORCE_CONSTANTS,NEIGHBOUR_DIRECTIONS,MC_CYCLES,MC_TEMP,neighbourContacts,atoms)

