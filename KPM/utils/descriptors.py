import numpy as np
from rdkit.Chem import AllChem, rdmolops
from rdkit import DataStructs
from numpy.typing import ArrayLike

def calc_fps(num_reacs: int, desc_type: str, desc_construction: str,
             rmol: list, pmol: list, dH: ArrayLike, 
             morgan_radius: int = 5, morgan_num_bits: int = 1024):
    '''Calculate reaction fingerprints.
    
    Capable of calculating difference, concatenation or
    difference+concatenation  fingerprints from
    SOAP, MBTR and Morgan fingerprint descriptors. Adds
    reaction enthalpy change at the end of the fingerprint
    as an extra feature.

    Arguments:
        num_reacs: Number of reactions to process difference fingerprints for.
        desc_type: One of ['soap', 'mbtr', 'MorganF'].
        desc_construction: One of ['diffs', 'concat', 'diffs+concat'].
        rmol: List of reactant RDKit MOL objects.
        pmol: List of product RDKit MOL objects.
        dH: Array of reaction enthalpy changes.
        morgan_radius: (Optional; Default = 5) Radius of Morgan Fingerprint (if using).
        morgan_num_bits: (Optional; Default: 1024) Size (in bits) of hashed Morgan fingerprint (if using).
    '''
    if desc_type == 'soap':
        fps = calc_fps_soap()
    elif desc_type == 'mbtr':
        fps = calc_fps_mbtr()
    elif desc_type == 'MorganF':
        fps = calc_fps_morgan(num_reacs, desc_construction, 
                              rmol, pmol, dH, morgan_radius, 
                              morgan_num_bits)
    else:
        raise ValueError('Unknown desc_type!')

    return fps


def calc_fps_soap():
    raise NotImplementedError()


def calc_fps_mbtr():
    raise NotImplementedError()


def calc_fps_morgan(num_reacs: int, dc: str, rmol: list, pmol: list, dH: ArrayLike, 
                    radius: int, num_bits: int):
    '''Calculate Morgan fingerprint-based reaction fingerprints.
    '''
    dc_featmap = {
        'diffs': 1,
        'concat': 2,
        'diffs+concat': 3
    }
    fps = np.zeros([num_reacs, (num_bits*dc_featmap[dc])+1])
    for i in range(num_reacs):

        # Calculate Morgan Fingerprint counts for reactants and products.
        fp1 = AllChem.GetHashedMorganFingerprint(rmol[i], radius, nBits=num_bits)
        fp2 = AllChem.GetHashedMorganFingerprint(pmol[i], radius, nBits=num_bits)  

        # Convert fingerprint count arrays to numpy arrays.
        fp_arr1 = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp1, fp_arr1)
        fp_arr2 = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp2, fp_arr2)

        # Construct final fingerprint.
        if dc == 'diffs':
            # Calculate difference.
            for j in range(num_bits):
                fps[i, j] = fp_arr2[j] - fp_arr1[j]     
                
        elif dc == 'concat':
            for j in range(num_bits):
                fps[i, j] = fp_arr1[j]
                fps[i, num_bits+j] = fp_arr2[j]

        elif dc == 'diffs+concat':
            for j in range(num_bits):
                fps[i, j] = fp_arr1[j]
                fps[i, num_bits+j] = fp_arr2[j] - fp_arr1[j]
                fps[i, 2*num_bits+j] = fp_arr2[j]

        else:
            raise ValueError('Unknown value for descriptor_construction. How did you get here?')

        # Add dH to the descriptor vector.
        fps[i, -1] = dH[i]

    return fps


def get_atypes(rmol: list):
    '''Gets a list of all unique atom types.
    
    Arguments:
        rmol: List of reactant RDKit MOL objects (equivalent to products in terms of atom types).'''
    atypes = set()
    for mol in rmol:
        for atom in mol.GetAtoms():
            atypes.add(atom.GetSymbol())

    return atypes

def calc_natom_features(num_reacs: int, rmol: list, pmol: list, atypes: list):
    '''Calculate additional features relating to numbers of atoms in reactions.

    Currently implements (for each atom type in `atypes`):
    * Number of atoms.
    * Total number of positive charges on atoms in reactants and products.
    * Total number of negative charges on atoms in reactants and products.
    * Total number of radical electrons on atoms in reactants and products.
    * Total number of molecules in reactants and products.

    Arguments:
        num_reacs: Number of reactions to process difference fingerprints for.
        rmol: List of reactant RDKit MOL objects.
        pmol: List of product RDKit MOL objects.
        atypes: List of atom types in training dataset.
    '''
    features = np.zeros((num_reacs, len(atypes)*7 + 2), dtype=int)
    atypes_sorted = sorted(list(atypes))
    for i in range(num_reacs):
        atype_dict = {at: 0 for at in atypes_sorted}
        reac_pos_chg_dict = {at: 0 for at in atypes_sorted}
        prod_pos_chg_dict = {at: 0 for at in atypes_sorted}
        reac_neg_chg_dict = {at: 0 for at in atypes_sorted}
        prod_neg_chg_dict = {at: 0 for at in atypes_sorted}
        reac_rad_dict = {at: 0 for at in atypes_sorted}
        prod_rad_dict = {at: 0 for at in atypes_sorted}
        for atom in rmol[i].GetAtoms():
            sym = atom.GetSymbol()
            atype_dict[sym] += 1
            chg = atom.GetFormalCharge()
            if chg > 0:
                reac_pos_chg_dict[sym] += chg
            elif chg < 0:
                reac_neg_chg_dict[sym] += chg
            reac_rad_dict[sym] += atom.GetNumRadicalElectrons()
        for atom in pmol[i].GetAtoms():
            sym = atom.GetSymbol()
            chg = atom.GetFormalCharge()
            if chg > 0:
                prod_pos_chg_dict[sym] += chg
            elif chg < 0:
                prod_neg_chg_dict[sym] += chg
            prod_rad_dict[sym] += atom.GetNumRadicalElectrons()
        
        atype_feats = np.array([atype_dict[at] for at in atypes_sorted])
        chg_feats = np.concatenate([
            [reac_pos_chg_dict[at] for at in atypes_sorted],
            [reac_neg_chg_dict[at] for at in atypes_sorted],
            [prod_pos_chg_dict[at] for at in atypes_sorted],
            [prod_neg_chg_dict[at] for at in atypes_sorted]])
        rad_feats = np.concatenate([
            [reac_rad_dict[at] for at in atypes_sorted],
            [prod_rad_dict[at] for at in atypes_sorted]])
        nmol_feats = [len(rdmolops.GetMolFrags(pmol[i])), len(rdmolops.GetMolFrags(rmol[i]))]
        features[i, :] = np.concatenate((atype_feats, chg_feats, rad_feats, nmol_feats))

    return features
