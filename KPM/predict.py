'''KPM Activation Energy Prediction Module

Used to predict activation energy (Eact) of reactions
from existing KPM-trained models.
'''

import numpy as np
from openbabel import pybel
from rdkit import Chem
from KPM.utils.descriptors import calc_diffs
from KPM.utils.data_funcs import un_normalise


class ModelPredictor:
    '''Uses a previously trained KPM model to predict Eact for the given reaction.
    
    Converts the input rection defined by a reactant/product
    pair into a reaction difference fingerprint of the same
    kind used in training the respective model, then runs
    activation energy prediction for this reaction. 

    If outfile is not defined, will simply print the prediction
    to stdout. If outfile is defined, will also place this
    prediction in the specified file.

    Arguments:
        args: argparse Namespace from CLI.
    '''
    def __init__(self, args):
        '''Initialise class from supplied CLI arguments.'''

        print('--------------------------------------------')
        print('KPM Model Activation Energy Prediction')

        # Sort out prediction arguments first.
        self.argparse_args = args

        self.model = args.model
        print(f'Using model in {self.model}')
        self.reac = args.reactant
        self.prod = args.product
        self.enthalpy = args.enthalpy
        self.outfile = args.outfile
        self.verbose = True if args.verbose == 'True' else False

        if self.outfile is not None:
            print(f'Prediction output will be saved to {self.outfile}')

        # Load in existing KPM model.
        model = np.load(self.model, allow_pickle=True)
        self.regr = model['models']
        self.scaler = model['scaler'][()] # scaler gets saved as a 0D array.
        norm_vars = model['norm_vars']
        self.norm_avg_Eact = norm_vars[0]
        self.norm_std_Eact = norm_vars[1]
        self.training_args = model['args'][()] # args gets saved as a 0D array.
        args = self.training_args

        self.nn_ensemble_size = args.nn_ensemble_size
        self.descriptor_type = args.descriptor_type
        self.norm_type = args.norm_type
        self.morgan_num_bits = args.morgan_num_bits
        self.morgan_radius = args.morgan_radius

        print('--------------------------------------------\n')


    def get_smiles_from_xyz(self):
        '''Turn an xyz file into a SMILES string.
        
        Turns xyz structure into an OpenBabel OBMol object, then
        extracts SMILES string from this.
        '''
        rgen = pybel.readfile('xyz', self.reac)
        rmol = []
        gen_stat = True
        while gen_stat:
            try:
                rm = next(rgen)
                rmol.append(rm)
            except StopIteration:
                gen_stat = False

        pgen = pybel.readfile('xyz', self.prod)
        pmol = []
        gen_stat = True
        while gen_stat:
            try:
                pm = next(pgen)
                pmol.append(pm)
            except StopIteration:
                gen_stat = False

        # These functions return the SMILES followed by the xyz file path, hence the split/strip.
        rsmi = [mol.write('can').split()[0].strip() for mol in rmol]
        psmi = [mol.write('can').split()[0].strip() for mol in pmol]

        return rsmi, psmi


    def process_xyzs(self):
        '''Turn reactant/product xyzs into a difference fingerprint.'''
        # Transform xyzs into SMILES.
        rsmi, psmi = self.get_smiles_from_xyz()
        self.rsmi = rsmi
        self.psmi = psmi

        # Transform SMILES into RDKit MOL objects.
        rmol = [Chem.MolFromSmiles(smi) for smi in rsmi]
        pmol = [Chem.MolFromSmiles(smi) for smi in psmi]

        # Load in enthalpies from file.
        self.dH_arr = np.loadtxt(self.enthalpy)

        # Check lengths of array/lists.
        if len(self.dH_arr) != len(rmol) or len(self.dH_arr) != len(pmol):
            raise RuntimeError('Input data has different lengths!')
        else:
            self.num_reacs = len(self.dH_arr)

        # Calculate reaction difference fingerprint.
        diffs = calc_diffs(self.num_reacs, self.descriptor_type, rmol, pmol, self.dH_arr,
                          self.morgan_radius, self.morgan_num_bits)

        return diffs

    
    def predict(self, diffs):
        '''Predict Eacts from difference fingerprints.
        
        Arguments:
            diffs: Reaction difference fingerprints.
        '''
        if self.verbose: print('Getting reaction difference fingerprints.')
        diffs = self.scaler.transform(diffs)

        if self.verbose: print(f'Predicting activation energy over {self.nn_ensemble_size} NNs.')
        Eact_pred = np.zeros(self.num_reacs)
        for i in range(self.nn_ensemble_size):
            pred = self.regr[i].predict(diffs)
            pred = un_normalise(pred, self.norm_avg_Eact, self.norm_std_Eact, self.norm_type)
            Eact_pred += pred

        Eact_pred = Eact_pred / self.nn_ensemble_size

        if self.outfile is not None:
            output = [
                '# KPM Eact Prediction',
                f'# Reactant File: {self.reac}',
                f'# Product File: {self.prod}',
                '\n'
            ]
            with open(self.outfile, 'w') as f:
                f.writelines('\n'.join(output))

        for i in range(self.num_reacs):
            print(f'Reaction {i+1}: Predicted Eact = {Eact_pred[i]} kcal/mol')
            if self.outfile is not None:

                output = [
                    f'# Reaction {i+1}',
                    f'# Reactant SMILES: {self.rsmi[i]}',
                    f'# Product SMILES: {self.psmi[i]}',
                    f'# dH: {self.dH_arr[i]} kcal/mol',
                    f'# Eact prediction (in kcal/mol) follows:',
                    f'{Eact_pred[i]}',
                    '\n'
                ]
                with open(self.outfile, 'a') as f:
                    f.writelines('\n'.join(output))
                
        if self.verbose: print('Output written to file.')