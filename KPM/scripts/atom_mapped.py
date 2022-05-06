'''This script was extracted from ard_gsm https://github.com/cgrambow/ard_gsm
'''

from molg import*
import csv
#from PACKS import*
#from FUNCTIONS import*
#from NewModules import*
from NewModules import*
import os
import glob
import argparse
import sys
import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--directory', type=str, action='store')
    args = parser.parse_args()
    if (args.directory == None):
        print("**You have NOT entered a command line argument for the directory name\n")
    else:
        rr, pp = atommap(args.directory)

def atommap(directory):
    """
    This function converts an xyz file format to atom-mapped smiles to match the training data smiles format.
    Here we specify a directory. Reactant and product xyzs are assumed to be stored in directories
    called ./reactants/ and ./products/. The outputs are stored in a file called canonical_smiles.csv in the specified
    directory. This code can replace the previously used openbabel.
    """

    rxn_files  = []
    prod_files = []
    for file in sorted(glob.glob(directory + './reactants/*.xyz'),key=numericalSort):
        rxn_files.append(file)
    for file in sorted(glob.glob(directory + './products/*.xyz'),key=numericalSort):
        prod_files.append(file)
    
    nrx=len(rxn_files)
    c=1   
    smiles_r = []
    smiles_p = []
 
    for i in range(nrx):
    
       dat = read_xyz_file(rxn_files[i])[0]#*.xyz')
       symbols, coords = dat	
       #print(dat)
       rr = MolGraph(symbols=symbols, coords=coords)
       rr.infer_connections()
       dat = read_xyz_file(prod_files[i])[0]#*.xyz')
    
       symbols, coords = dat
       #print(dat)
       pp = MolGraph(symbols=symbols, coords=coords)
       pp.infer_connections()
       try:
        smilr = rr.perceive_smiles()
        smiles_r.append(smilr)
#        print(c,"***reactant smiles: ",smilr)
       except ValueError:
        continue
       if smilr is None:
        continue
       rr = Chem.MolFromSmiles(smilr)
       try:
        smilp = pp.perceive_smiles()
        smiles_p.append(smilp)
#        print(c,"***product smiles: ",smilp)
       except ValueError:
        continue
       if smilp is None:
        continue
       pp = Chem.MolFromSmiles(smilp)
       c+=1
    with open(directory + 'canonical_smiles.csv', 'w', ) as myfile:
        wr = csv.writer(myfile, delimiter=',')
        header = ['idx','rsmi','psmi']
        wr.writerow(header)
        for i in range(nrx):
            wr.writerow([i,smiles_r[i],smiles_p[i]])

    return rr, pp

#rr, pp = atommap('./UnimolecularRXN/') 

if __name__ == '__main__':
    main()
