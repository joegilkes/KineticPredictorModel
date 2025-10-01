'''KPM Activation Energy Prediction Module

Used to predict activation energy (Eact) of reactions
from existing KPM-trained models.
'''

OBCR_enabled = True
try:
    from obcr import fix_radicals
    from obcr.utils import pbmol_to_smi, is_radical
except ImportError:
    OBCR_enabled = False

from collections import Counter
import numpy as np
from openbabel import pybel
from rdkit import Chem, RDLogger
from KPM.utils.descriptors import calc_fps, calc_natom_features
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
        self.direction = args.direction
        self.do_uncertainty = True if args.uncertainty == 'True' else False
        self.verbose = True if args.verbose == 'True' else False

        if args.fix_radicals == 'True':
            if not OBCR_enabled:
                _errstr = (
                    'Argument \'--fix_radicals\' requires OBCanonicalRadicals to be installed.\n'
                    'Follow the installation instructions for obtaining and installing the submodule.'  
                )
                raise ModuleNotFoundError(_errstr)
            else:
                print('Canonical radical structure fixing via OBCanonicalRadicals is enabled.')
                self.fix_radicals = True
        else:
            self.fix_radicals = False

        if args.suppress_rdlogs == 'True':
            RDLogger.DisableLog('rdApp.*')

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
        self.norm_eacts = True if args.norm_eacts == 'True' else False
        self.morgan_num_bits = args.morgan_num_bits
        self.morgan_radius = args.morgan_radius

        # Newer features with defaults for compatibility with older models.
        if 'use_natom_features' in args._get_args():
            self.use_natom_features = args.use_natom_features
            self.atypes = args.atypes
        else:
            self.use_natom_features = False
            self.atypes = set()
        if 'descriptor_construction' in args._get_args():
            self.descriptor_construction = args.descriptor_construction
        else:
            self.descriptor_construction = 'diffs'

        if self.do_uncertainty and self.nn_ensemble_size == 1:
            raise ValueError('Prediction uncertainty cannot be calculated with only a single model in the ensemble.')

        print('--------------------------------------------\n')


    def remove_spectators(self, rsmi_list, psmi_list):
        '''Remove spectator molecules from a list of reactions.

        Iterates through reactant-product system pairs and identifies
        molecules which are not taking part in any changes in bonding.
        These will be effectively removed anyway by the subtraction of
        descriptors, but this greatly speeds up the canonicalisation of
        radical structures if OBCR is used.

        Returns a mask, indicating any reactions that have been removed 
        entirely due to no reaction occurring with a 0.
        '''
        final_rsmi_list = []
        final_psmi_list = []
        mask = [True for _ in range(len(rsmi_list))]
        for i, (rsmi, psmi) in enumerate(zip(rsmi_list, psmi_list)):
            rsep = Counter(rsmi.split('.'))
            psep = Counter(psmi.split('.'))
            # Skip over this reaction if reacs and prods are exactly the same.
            if rsep == psep:
                mask[i] = False
                continue

            for reac in rsep:
                while rsep[reac] > 0:
                    if reac in psep.keys() and psep[reac] > 0:
                        rsep[reac] -= 1
                        psep[reac] -= 1
                    else:
                        break

            rsmi_final = '.'.join(['.'.join([r for _ in range(rsep[r])]) for r in rsep if rsep[r] > 0])
            psmi_final = '.'.join(['.'.join([p for _ in range(psep[p])]) for p in psep if psep[p] > 0])
            print(rsmi_final, psmi_final)
            final_rsmi_list.append(rsmi_final)
            final_psmi_list.append(psmi_final)

        return final_rsmi_list, final_psmi_list, mask


    def fix_radical_list(self, smi_list):
        '''Fix a list of SMILES stings with OBCanonicalRadicals.

        Iterates through a list of input SMILES strings, creating
        their respective `pybel.Molecule`s and separating any fragment
        species. Cleans up/canonicalises the radical structure of
        and relevant species, then combines them back into a single
        SMILES and returns.
        '''
        final_smi_list = []
        for smi in smi_list:
            pbmol = pybel.readstring('smi', smi)
            pbmol.addh()
            pbmol.make3D()
            species_list = [pybel.Molecule(obmol) for obmol in pbmol.OBMol.Separate()]

            smi_list = []
            for species in species_list:
                spec_smi = pbmol_to_smi(species)
                if is_radical(spec_smi):
                    print(f'Initial species SMILES: {spec_smi}')
                    species = fix_radicals(species)
                    # Force re-parsing of structure to ensure aromaticity is detected.
                    species.addh()
                    spec_smi = pbmol_to_smi(species)
                    print(f'Final species SMILES: {spec_smi}')
                    smi_list.append(spec_smi)
                else:
                    smi_list.append(spec_smi)

            if len(smi_list) == 1:
                final_smi = smi_list[0]
            else:
                final_smi = '.'.join(smi_list)

            final_smi_list.append(final_smi)

        return final_smi_list


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

        # Remove non-participating molecules from lists.
        rsmi, psmi, mask = self.remove_spectators(rsmi, psmi)

        # Tidy up weird radical structures so all radicals are consistent, if requested.
        if self.fix_radicals:
            rsmi = self.fix_radical_list(rsmi)
            psmi = self.fix_radical_list(psmi)

        return rsmi, psmi, mask


    def process_xyzs(self):
        '''Turn reactant/product xyzs into a difference fingerprint.'''
        # Transform xyzs into SMILES.
        rsmi, psmi, mask = self.get_smiles_from_xyz()
        self.rsmi = rsmi
        self.psmi = psmi

        # Transform SMILES into RDKit MOL objects.
        rmol = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in rsmi]
        pmol = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in psmi]

        # Load in enthalpies from file.
        self.dH_arr = np.loadtxt(self.enthalpy, ndmin=1)

        # Modify dH array to respect null reactions from mask.
        self.dH_arr = self.dH_arr[mask]

        # Check lengths of array/lists.
        if len(self.dH_arr) != len(rmol) or len(self.dH_arr) != len(pmol):
            raise RuntimeError('Input data has different lengths!')
        else:
            self.num_reacs = len(self.dH_arr)

        # Manipulate arrays based on prediction direction.
        if self.direction == 'backward':
            self.rsmi, self.psmi = self.psmi, self.rsmi
            rmol, pmol = pmol, rmol
            self.dH_arr = np.flip(self.dH_arr)
        elif self.direction == 'both':
            rsmi_combined = []
            psmi_combined = []
            rmol_combined = []
            pmol_combined = []
            dH_combined = np.zeros(self.num_reacs*2)
            for i, (rs, ps, rm, pm) in enumerate(zip(rsmi, psmi, rmol, pmol)):
                rsmi_combined.append(rs)
                psmi_combined.append(ps)
                rmol_combined.append(rm)
                pmol_combined.append(pm)
                dH_combined[2*i] = self.dH_arr[i]
                rsmi_combined.append(ps)
                psmi_combined.append(rs)
                rmol_combined.append(pm)
                pmol_combined.append(rm)
                dH_combined[(2*i)+1] = -self.dH_arr[i]
            self.rsmi = rsmi_combined
            self.psmi = psmi_combined
            rmol = rmol_combined
            pmol = pmol_combined
            self.dH_arr = dH_combined
        
        n_reacs_adj = self.num_reacs if self.direction != 'both' else self.num_reacs*2

        # Calculate reaction difference fingerprint.
        fps = calc_fps(n_reacs_adj, self.descriptor_type, self.descriptor_construction,
                        rmol, pmol, self.dH_arr, self.morgan_radius, self.morgan_num_bits)
        
        if self.use_natom_features:
            nafs = calc_natom_features(n_reacs_adj, rmol, pmol, self.atypes)
            fps = np.concatenate((fps, nafs), axis=1)

        return fps

    
    def predict(self, fps):
        '''Predict Eacts from difference fingerprints.
        
        Arguments:
            fps: Reaction difference fingerprints.
        '''
        if self.verbose: print('Getting reaction difference fingerprints.')
        fps = self.scaler.transform(fps)

        if self.verbose: print(f'Predicting activation energies over {self.nn_ensemble_size} NNs.')
        n_reacs_adj = self.num_reacs if self.direction != 'both' else self.num_reacs*2
        Eact_pred = np.zeros((n_reacs_adj, self.nn_ensemble_size))
        for i in range(self.nn_ensemble_size):
            pred = self.regr[i].predict(fps)
            if self.norm_eacts:
                pred = un_normalise(pred, self.norm_avg_Eact, self.norm_std_Eact, self.norm_type)
            Eact_pred[:, i] = pred

        Eacts = np.mean(Eact_pred, axis=1)
        uncerts = np.std(Eact_pred, axis=1)

        if self.outfile is not None:
            output = [
                '# KPM Eact Prediction',
                f'# Reactant File: {self.reac}',
                f'# Product File: {self.prod}',
                '\n'
            ]
            with open(self.outfile, 'w') as f:
                f.writelines('\n'.join(output))

        def uncert_str(i):
            if self.do_uncertainty:
                return f' ± {uncerts[i]}'
            else:
                return ''

        for i in range(self.num_reacs):
            if self.direction == 'forward':
                print(f'Reaction {i+1}: Predicted Eact = {Eacts[i]}{uncert_str(i)} kcal/mol')
                if self.outfile is not None:

                    output = [
                        f'# Reaction {i+1}',
                        f'# Reactant SMILES: {self.rsmi[i]}',
                        f'# Product SMILES: {self.psmi[i]}',
                        f'# dH: {self.dH_arr[i]} kcal/mol',
                        f'# Eact prediction (in kcal/mol) follows:',
                        f'{Eacts[i]}{uncert_str(i)}',
                        '\n'
                    ]
                    with open(self.outfile, 'a') as f:
                        f.writelines('\n'.join(output))

            elif self.direction == 'backward':
                print(f'Backward Reaction {i+1}: Predicted Eact = {Eacts[i]}{uncert_str(i)} kcal/mol')
                if self.outfile is not None:

                    output = [
                        f'# Backward Reaction {i+1}',
                        f'# Reactant SMILES: {self.rsmi[i]}',
                        f'# Product SMILES: {self.psmi[i]}',
                        f'# dH: {self.dH_arr[i]} kcal/mol',
                        f'# Eact prediction (in kcal/mol) follows:',
                        f'{Eacts[i]}{uncert_str(i)}',
                        '\n'
                    ]
                    with open(self.outfile, 'a') as f:
                        f.writelines('\n'.join(output))

            elif self.direction == 'both':
                print(f'Forward Reaction {i+1}: Predicted Eact = {Eacts[2*i]}{uncert_str(2*i)} kcal/mol')
                if self.outfile is not None:

                    output = [
                        f'# Forward Reaction {i+1}',
                        f'# Reactant SMILES: {self.rsmi[2*i]}',
                        f'# Product SMILES: {self.psmi[2*i]}',
                        f'# dH: {self.dH_arr[2*i]} kcal/mol',
                        f'# Eact prediction (in kcal/mol) follows:',
                        f'{Eacts[2*i]}{uncert_str(2*i)}',
                        '\n'
                    ]
                    with open(self.outfile, 'a') as f:
                        f.writelines('\n'.join(output))

                print(f'Backward Reaction {i+1}: Predicted Eact = {Eacts[2*i+1]}{uncert_str(2*i+1)} kcal/mol')
                if self.outfile is not None:

                    output = [
                        f'# Backward Reaction {i+1}',
                        f'# Reactant SMILES: {self.rsmi[2*i+1]}',
                        f'# Product SMILES: {self.psmi[2*i+1]}',
                        f'# dH: {self.dH_arr[2*i+1]} kcal/mol',
                        f'# Eact prediction (in kcal/mol) follows:',
                        f'{Eacts[2*i+1]}{uncert_str(2*i+1)}',
                        '\n'
                    ]
                    with open(self.outfile, 'a') as f:
                        f.writelines('\n'.join(output))
                
        if self.verbose: print('Output written to file.')

        return Eacts, uncerts