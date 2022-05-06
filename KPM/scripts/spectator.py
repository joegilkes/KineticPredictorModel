# Routine to calculate connectivity matrix of ASE object...
import ase
from ase.io import read
import numpy as np
from ase import Atoms
from NewModules import*
def GetGraph(cstr):
    """
    This subroutine takes an ASE object called cstr and calculates the connectivity (adjacency) matrix using the
    same definitions as used in the original CDE code.
    """

    # Get the list of atom symbols
    atsym = cstr.get_chemical_symbols()

    # Set number of atoms
    n = len(atsym)

    # Get positions
    positions = cstr.get_positions()

    # Zero-out the graph.
    graph = np.zeros([n,n],dtype='int')

    # Set up a dictionary of covalent radii....in Angstroms.
    # NOTE: You'll need to add elements if they're not here!!!
    CovalentRadii = { 'H' : 0.31,
                  'C' : 0.76,
                  'N' : 0.71,
                  'O' : 0.66
                }
    BondingScaleFactor = 1.2

    for i in range(n):
        radius_i = CovalentRadii[atsym[i]]
        for j in range(i+1,n):
            radius_j = CovalentRadii[atsym[j]]
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dz = positions[i][2] - positions[j][2]
            distance = np.sqrt(dx*dx+dy*dy+dz*dz)
            rcut = (radius_i + radius_j) * BondingScaleFactor
            if distance <= rcut:
                graph[i,j] = 1
                graph[j,i] = 1

    return graph

def numericalSort(value):

    """
    Routine to return the numerical part of a string or filename - used for sorting files.
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


import glob
#Load  Dataset
reactants_dir = "./KineticPredictorModel/KPM/Data/Test/Aldol/rxn/"
products_dir = "./KineticPredictorModel/KPM/Data/Test/Aldol/pxn/"
# Read Reactant structure files in order
reactant_files = []
for file in sorted(glob.glob(reactants_dir+"/react_P*.xyz"),key=numericalSort):
    reactant_files.append(file)
#print(reactant_files)    

product_files = []
for file in sorted(glob.glob(products_dir+"/prod_P*.xyz"),key=numericalSort):
    product_files.append(file)
#print(product_files)    

#Convert all to ASE atoms object
reactant_struct = []
for file in reactant_files:   
    reactant_struct.append(ase.io.read(file,index=':',format='xyz'))

product_struct = []
for file in product_files:   
    product_struct.append(ase.io.read(file,index=':',format='xyz'))

# print(product_struct)    
#Define reactant and product frames
#Select reactant and product frames for benzene dataset    
product = []
reactant = []
nreact = len(reactant_files)
for i in range(nreact):
    product.append(product_struct[i])
    reactant.append(reactant_struct[i])
#print(product)

reactant = [ r[0] for r in reactant ]
product = [ r[0] for r in product ]

#Organics- removing non-reactive elements 
# nreact = len(struct2)
print(nreact)

gp=[]
gr=[]
for i in range(nreact):
    gr.append(GetGraph(reactant[i]))
    gp.append(GetGraph(product[i]))
#print(gr)
#print(gp)
##Organics- removing non-reactive elements 

nreact = len(reactant_struct)

for i in range(nreact):
    #reactive_atoms=[]
    n = len(reactant[i].get_chemical_symbols())
    reactant[i].rxtv = np.zeros(n, dtype='bool')
    product[i].rxtv = np.zeros(n, dtype='bool')

    reactant[i].rxid = []
    product[i].rxid = []
    reactant[i].rxidpair = []
    graph_P = GetGraph(product[i])
    graph_R = GetGraph(reactant[i])
    dG = graph_P - graph_R
    print(dG)
    '''
    Here you want to get rid of spectator atoms.
    i.e. if the atom in reactant i does change in the product 
    ~ not equal to zero, append, else get rid of.
    ''' 
    for j in range(n):
        if any(dG[j, :]):
            reactant[i].rxtv[j] = True
            reactant[i].rxid.append(j)
            product[i].rxtv[j] = True
            product[i].rxid.append(j)
    for j in range(n - 1):
        for k in range(j + 1, n):
            if dG[j, k] != 0:
                reactant[i].rxidpair.append([j, k])
   # Test print of positions for a chosen structure...
#
#print(struct[0][0].get_positions())
print("REACTANT ",reactant[0].get_chemical_symbols())
print("PRODUCT ",product[0].get_chemical_symbols()) 
    #reactive_atoms=[]
#    n = len(reactant[i].get_chemical_symbols())
#    reactant[i].rxtv = np.zeros(n, dtype='bool')
#    product[i].rxtv = np.zeros(n, dtype='bool')
#
#    reactant[i].rxid = []
#    product[i].rxid = []
#    reactant[i].rxidpair = []
#    graph_P = GetGraph(product[i])
#    graph_R = GetGraph(reactant[i])
#    dG = graph_P - graph_R
#    gp.append(graph_P)

    # for j in range(n):
    #     if any(dG[j, :]):
    #         reactant[i].rxtv[j] = True
    #         reactant[i].rxid.append(j)
    #         product[i].rxtv[j] = True
    #         product[i].rxid.append(j)
    # for j in range(n - 1):
    #     for k in range(j + 1, n):
    #         if dG[j, k] != 0:
    #             reactant[i].rxidpair.append([j, k])
