from _elements import elements_, numbers_
import numpy as np

def P1cifHeader(file, a,b,c,al,be,ga):
    with open(file,"w") as file:
        file.write("_cell_length_a                         "+str(a)+"\n")
        file.write("_cell_length_b                         "+str(b)+"\n")
        file.write("_cell_length_c                         "+str(c)+"\n")
        file.write("_cell_angle_alpha                      "+str(al)+"\n")
        file.write("_cell_angle_beta                       "+str(be)+"\n")
        file.write("_cell_angle_gamma                      "+str(ga)+"\n")
        file.write("_space_group_name_H-M_alt              'P 1' \n")
        file.write("_space_group_IT_number                 1 \n  \n")
        file.write("loop_        \n")
        file.write("_space_group_symop_operation_xyz \n")
        file.write("   'x, y, z'    \n     \n")
        file.write("loop_ \n")
        file.write("   _atom_site_type_symbol      \n")
        file.write("   _atom_site_fract_x               \n")
        file.write("   _atom_site_fract_y               \n")
        file.write("   _atom_site_fract_z               \n") 
    file.close() 

def FullAsCif(filestem,run,CELL,BOX_SIZE,SITES,atomicModel,repeats):
    # Print out all the full atomic model, including padding unit cells as a cif
    # ONLY USE THIS IF THE SUPERCELL IS SMALL!
    P1cifHeader(filestem+"_atoms_0"+f"{run:01d}_Full.cif",CELL[0]*BOX_SIZE[0],CELL[1]*BOX_SIZE[1],CELL[2]*BOX_SIZE[2],CELL[3],CELL[4],CELL[5])
    nx = BOX_SIZE[0]
    ny = BOX_SIZE[1]
    nz = BOX_SIZE[2]
    natoms = len(atomicModel)
    
    with open(filestem+"_atoms_0"+f"{run:01d}_Full.cif","a") as file2:       
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for l in range(SITES):
                        address = int(((i*(ny)*(nz))+(j*(nz))+k)*natoms + l)
                        file2.write(numbers_[int(atomicModel[address,1])]+"    {:.6f}".format((atomicModel[address,2]+i)/(nx))+ "  {:.6f}".format((atomicModel[address,3]+j)/(ny))+"  {:.6f}  ".format((atomicModel[address,4]+k)/(nz))+"  \n")
    print("     full CIF")      
    file2.close() 

def Average(filestem,run, average,modelParams,Uiso):
    # print a cif of the average structure
    a = modelParams.a
    b = modelParams.b
    c = modelParams.c
    al = modelParams.al
    be = modelParams.be
    ga = modelParams.ga
    natoms = modelParams.natoms

    P1cifHeader(filestem+"_atoms_"+f"{run:02d}_average.cif",a,b,c,al,be,ga)
    with open(filestem+"_atoms_"+f"{run:02d}_average.cif","a") as file2:
        file2.write("    _atom_site_occupancy\n")
        file2.write("    _atom_site_U_iso_or_equiv\n")
        for l in range(natoms):
            if average[l,6] != 0:
                file2.write(numbers_[int(average[l,4])]+"  {:.6f}".format(average[l,1])+ "  {:.6f}".format(average[l,2])+"  {:.6f}".format(average[l,3])+ "   {:.6f}".format(average[l,5]*average[l,0])+"   {:.6f}".format(Uiso[l])+"\n")   
                file2.write(numbers_[int(average[l,6])]+"  {:.6f}".format(average[l,1])+ "  {:.6f}".format(average[l,2])+"  {:.6f}".format(average[l,3])+ "   {:.6f}".format(average[l,7]*average[l,0])+"   {:.6f}".format(Uiso[l])+"\n")   
            else: file2.write(numbers_[int(average[l,4])]+"  {:.6f}".format(average[l,1])+ "  {:.6f}".format(average[l,2])+"  {:.6f}".format(average[l,3])+ "   {:.6f}".format(average[l,0])+"   {:.6f}".format(Uiso[l])+"\n")    
    print("     average CIF")   
    file2.close() 

def Scatty(filestem,run,atomsArray,average,CELL,BOX_SIZE,SITE_ATOMS,USE_PADDING):
    # print the atomic model to a format that can be read by Scatty - a program used for calculating the diffuse scattering
    a = CELL[0]
    b = CELL[1]
    c = CELL[2]
    al = CELL[3]
    be = CELL[4]
    ga = CELL[5]
    natoms = sum(SITE_ATOMS)
    nx = BOX_SIZE[0]
    ny = BOX_SIZE[1]
    nz = BOX_SIZE[2]

    if USE_PADDING == True: padding = 1
    else: padding = 0
    #Li = np.array([
    #[3,	 3,	    0.00500,	0.25500,	0.25500],
    #[3,	 3,	    0.00500,	0.75500,	0.75500],
    #[3,	 3,	    0.25500,	0.00500,	0.25500],
    #[3,	 3,	    0.25500,	0.25500,	0.00500],
    #[3,	 3,	    0.25500,	0.50500,	0.75500],
    #[3,	 3,	    0.25500,	0.75500,	0.50500],
    #[3,	 3,	    0.50500,	0.25500,	0.75500],
    #[3,	 3,	    0.50500,	0.75500,	0.25500],
    #[3,	 3,	    0.75500,	0.00500,	0.75500],
    #[3,	 3,	    0.75500,	0.25500,	0.50500],
    #[3,	 3,	    0.75500,	0.50500,	0.25500],
    #[3,	 3,	    0.75500,	0.75500,	0.00500]])
    
    with open(filestem+"_atoms_"+f"{run:02d}.txt", "w") as file1:
        file1.write("TITLE test   \n")
        file1.write("CELL   "+str(a)+" "+str(b)+" "+str(c)+" "+str(al)+"  "+str(be)+"  "+str(ga)+"\n")
        file1.write("BOX   "+str(nx-2*padding)+" "+str(ny-2*padding)+" "+str(nz-2*padding)+"\n")
        for i in range(0, len(average)):
            file1.write("SITE  {:.6f}".format(average[i,1])+"  {:.6f}".format(average[i,2])+"  {:.6f}".format(average[i,3])+"  \n")
        #for i in range(len(Li)):
        #    file1.write("SITE  {:.6f}".format(Li[i,2])+"  {:.6f}".format(Li[i,3])+"  {:.6f}".format(Li[i,4])+"  \n")

        for i in range(0, len(average)):
            if average[i,6] != 0: file1.write("OCC    "+numbers_[int(average[i,4])]+"   {:.6f}".format(average[i,5]*average[i,0])+"  "+numbers_[int(average[i,6])]+"   {:.6f}".format(average[i,7]*average[i,0])+"\n")
            else: file1.write("OCC    "+numbers_[int(average[i,4])]+"   {:.6f}".format(average[i,0])+"\n")
        #for i in range(len(Li)):
        #    file1.write("OCC    "+numbers_[int(Li[i,0])]+"   1\n")
   
        for i in range(padding, nx-padding):
            for j in range(padding,ny-padding):
                for k in range(padding,nz-padding):
                    for l in range(natoms):
                        address = ((i*(ny)*(nz))+(j*(nz))+k)*natoms + l
                        if atomsArray[address,1] != 0: 
                            file1.write("ATOM  "+str(l+1) + "  "+str(i-padding)+"  "+str(j-padding)+"  "+str(k-padding)+"  {:.6f}".format(atomsArray[address,2]-average[l,1])+"  {:.6f}".format(atomsArray[address,3]-average[l,2])+"  {:.6f}".format(atomsArray[address,4]-average[l,3])+"  "+numbers_[atomsArray[address,1]]+"\n")
                    #for o in range(len(Li)):
                    #    file1.write("ATOM  "+str(natoms+o+1) + "  "+str(i-padding)+"  "+str(j-padding)+"  "+str(k-padding)+"  0.000000 0.000000 0.000000 Li\n")

    print("     Scatty")
    file1.close()

def Partial3D(filestem, supercell,a,b,c,nx,ny,nz,nsites,atoms,atomDict):
    ''''FIX'''
    # exports a pseudo structure to illustrate disordered configuration 
    P1cifHeader(filestem+"_Partial.cif",a*(nx),b*(ny),c*(nz),90,90,90)
    with open(filestem+"_Partial.cif","a") as file2:
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    for s in range(nsites):
                        siteAddress = (x*(ny)*(nz)+(y*(nz))+z)*nsites + s
                        site = atoms[f'S{s+1}_V{supercell[siteAddress]}'] 
                        siteCenter = site['a1']
                        if supercell[siteAddress] == 1: file2.write(atomDict[siteCenter[0]]+"    {:.6f}".format((siteCenter[1]+x)/(nx))+ "  {:.6f}".format((siteCenter[2]+y)/(ny))+"  {:.6f}  ".format((siteCenter[3]+z)/(nz))+"\n")
                        elif supercell[siteAddress] == -1: file2.write("C    {:.6f}".format((siteCenter[1]+x)/(nx))+ "  {:.6f}".format((siteCenter[2]+y)/(ny))+"  {:.6f}  ".format((siteCenter[3]+z)/(nz))+"\n")
    print("     partial CIF")

def PartialLayersMin(filestem,run,supercell):
    """FIX STACKING AXIS"""
    # exports a pseudo structure to illustrate disordered configuration 
    P1cifHeader(filestem+"_atoms_"+f"{run:02d}_Partial.cif",2,2*len(supercell),2,90,90,90)
    with open(filestem+"_atoms_"+f"{run:02d}_Partial.cif","a") as file2:
        for l in range(len(supercell)):
            if supercell[l] == 1: file2.write("S  0.5  {:.6f}".format((l)/(len(supercell)))+ "  0.5 \n")
            elif supercell[l] == -1: file2.write("C  0.5  {:.6f}".format((l)/(len(supercell)))+ "  0.5 \n")    
    print("     partial CIF")

def PartialLayers(filestem,supercell,modelParams,siteParams,atoms):
    a = modelParams.a
    b = modelParams.b
    c = modelParams.c
    nx = modelParams.nx
    ny = modelParams.ny
    nz = modelParams.nz
    nlayers = siteParams.nsites
    stackingAxis = siteParams.stackingAxis

    P1cifHeader(filestem+"_Partial.cif",a*(nx),b*(ny),c*(nz),90,90,90)
    with open(filestem+"_Partial.cif","a") as file2:
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    for l in range(nlayers):
                        axes = {0:x, 1:y, 2:z}
                        layerAddress = axes[stackingAxis]*nlayers + l
                        layer = atoms[f'L{l+1}_V{supercell[layerAddress]}'] 
                        layerCenter = layer['a1']
                        if supercell[layerAddress] == 1: file2.write(layerCenter[0]+"    {:.6f}".format((layerCenter[1]+x)/(nx))+ "  {:.6f}".format((layerCenter[2]+y)/(ny))+"  {:.6f}  ".format((layerCenter[3]+z)/(nz))+"\n")
                        elif supercell[layerAddress] == -1: file2.write("C    {:.6f}".format((layerCenter[1]+x)/(nx))+ "  {:.6f}".format((layerCenter[2]+y)/(ny))+"  {:.6f}  ".format((layerCenter[3]+z)/(nz))+"\n")
    print("     partial CIF")


def printFullModel(atomsArray,modelParams):
    nx = modelParams.nx
    ny = modelParams.ny
    nz = modelParams.nz
    natoms = modelParams.natoms
    types = []
    print("     Full model for Welberry")
    with open("model_crystal.dat", "w") as file1:
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    atomcounter = 1
                    for i in range(natoms):
                        address = ((x*ny*nz)+(y*nz)+z)*natoms + i
                        if atomsArray[address,1] not in types: types.append(atomsArray[address,1])
                        file1.write(str(x+1) +"   "+ str(y+1)+"  "+str(z+1)+"  1  "+str(atomcounter)+"  "+str(types.index(atomsArray[address,1])+1)+"  {:.6f}".format(atomsArray[address,2])+"  {:.6f}".format(atomsArray[address,3])+"  {:.6f}".format(atomsArray[address,4])+"  0.000  0.000  0.000\n")
                        atomcounter+=1
    print(types)                    
    file1.close()

def printForSpringy(atomsArray,modelParams):
    nx = modelParams.nx
    ny = modelParams.ny
    nz = modelParams.nz
    natoms = modelParams.natoms
    with open("springyCrystal.dat", "w") as file1:
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    for i in range(natoms):
                        address = ((x*ny*nz)+(y*nz)+z)*natoms + i
                        file1.write(str(int(atomsArray[address,1]))+ " {:.6f}".format(atomsArray[address,2])+"  {:.6f}".format(atomsArray[address,3])+"  {:.6f}".format(atomsArray[address,4])+"  "+str(x)+"  "+str(y)+"  "+str(z)+"  \n")
    

    file1.close()

def fold(filestem,atomsArray,modelParams,i,j,k,padding):
    a = i*modelParams.a
    b = j*modelParams.b
    c = k*modelParams.c
    nx = modelParams.nx
    ny = modelParams.ny
    nz = modelParams.nz
    natoms = modelParams.natoms
    print(len(atomsArray))

    P1cifHeader(filestem+"_folded.cif",a,b,c,90,90,90)
    with open(filestem+"_folded.cif","a") as file2:
        for x in range(padding, nx-padding):
            for y in range(padding, ny-padding):
                for z in range(padding, nz-padding):
                    for l in range(natoms):
                        address = ((x*ny*nz)+(y*nz)+z)*natoms + l
                        file2.write(numbers_[int(atomsArray[address,1])]+ " {:.6f}".format(((atomsArray[address,2]+x)*i)%1)+"  {:.6f}".format(((atomsArray[address,3]+y)*j)%1)+"  {:.6f}".format(((atomsArray[address,4]+z)*k)%1)+"  \n")
