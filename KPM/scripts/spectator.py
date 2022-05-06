# Routine to calculate connectivity matrix of ASE object...
import numpy as np
import ase
from ase.io import read
import glob


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



#Load  Dataset
reactants_dir = "./reactants/"
products_dir = "./products/"
# Read Reactant structure files in order
reactant_files = []
for file in sorted(glob.glob(reactants_dir+"/react_P*.xyz"),key=numericalSort):
    reactant_files.append(file)

product_files = []
for file in sorted(glob.glob(products_dir+"/prod_P*.xyz"),key=numericalSort):
    product_files.append(file)

#Convert all to ASE atoms object
reactant_struct = []
for file in reactant_files:
    reactant_struct.append(ase.io.read(file,index=':',format='xyz'))

product_struct = []
for file in product_files:
    product_struct.append(ase.io.read(file,index=':',format='xyz'))


#Define reactant and product frames

product = []
reactant = []
nreact = len(reactant_files)
for i in range(nreact):
    product.append(product_struct[i])
    reactant.append(reactant_struct[i])

reactant = [ r[0] for r in reactant ]
product = [ r[0] for r in product ]


graph = GetGraph(reactant) 

for i in range(nreact):

    n = len(reactant2[i].get_chemical_symbols())
    reactant2[i].rxtv = np.zeros(n, dtype='bool')
    product2[i].rxtv = np.zeros(n, dtype='bool')

    reactant2[i].rxid = []
    product2[i].rxid = []
    reactant2[i].rxidpair = []
    graph_P = GetGraph(product2[i])
    graph_R = GetGraph(reactant2[i])
    dG = graph_P - graph_R
    print(dG)
    for j in range(n):
        if any(dG[j, :]):
            reactant2[i].rxtv[j] = True
            reactant2[i].rxid.append(j)
            product2[i].rxtv[j] = True
            product2[i].rxid.append(j)
    for j in range(n - 1):
        for k in range(j + 1, n):
            if dG[j, k] != 0:
                reactant2[i].rxidpair.append([j, k])
