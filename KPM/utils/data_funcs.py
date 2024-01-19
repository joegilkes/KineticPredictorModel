import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from sklearn.model_selection import RepeatedKFold, train_test_split, ShuffleSplit
from sklearn.utils import shuffle

def load_dataset(csv_file: str, verbose: bool = False):
    '''Loads in a csv dataset of reactions.
    
    Dataset should be of the same format as defined by the
    supplied b97d3 dataset. This includes SMILES strings of
    reactants and products, the activation energy barriers and
    the enthalpies of the reactions.

    Arguments:
        csv_file: Path to csv dataset of reactions.
        verbose: Whether to print extra info.
    '''

    if verbose: print(f'Loading in dataset from {csv_file}')

    data = pd.read_csv(csv_file)
    ea = data.ea.tolist()              # Activation energies
    dh = data.dh.tolist()              # Enthalpy changes
    rs = data.rsmi.tolist()            # Reactant smiles
    ps = data.psmi.tolist()            # Product smiles

    if verbose: print('Dataset loaded.')

    return ea, dh, rs, ps


def extract_data(ea: list, dh: list, rs: list, ps: list, num_reacs: int, train_direction: str):
    '''Extract data from a dataset to work with a given train_direction.
    
    Takes the original data from the dataset and reorders it
    in the provided train direction, while also extracting
    RDKit MOL objects from the reactnat and product SMILES.

    'forward' train_direction simply extracts data in the
    order presented in the dataset, 'backward' extracts
    the reverse reactions of the ones in the dataset, and
    'both' extracts both the forward and backward reactions.

    Arguments:
        ea: Raw activation energies from dataset.
        dh: Raw enthalpy changes from dataset.
        rs: Reactant SMILES strings from dataset.
        ps: Product SMILES strings from dataset.
        num_reacs: Number of reactions in dataset (including train_direction).
        train_direction: Direction of reactions to train from.
    '''
    Eact = np.zeros(num_reacs)
    dH = np.zeros(num_reacs)
    rmol = []
    pmol = []

    smiles_params = rdmolfiles.SmilesParserParams()
    smiles_params.removeHs = False

    # Rework arrays based on train_direction
    if train_direction == 'forward':
        for i in range(num_reacs):
            Eact[i] = ea[i]
            dH[i] = dh[i]
            rmol.append(Chem.MolFromSmiles(rs[i], smiles_params))
            pmol.append(Chem.MolFromSmiles(ps[i], smiles_params))
    elif train_direction == 'backward':
        for i in range(num_reacs):
            Eact[i] = ea[i] - dh[i]
            dH[i] = -dh[i]
            rmol.append(Chem.MolFromSmiles(ps[i], smiles_params))
            pmol.append(Chem.MolFromSmiles(rs[i], smiles_params))
    elif train_direction == 'both':
        half_reacs = int(num_reacs/2)
        for i in range(half_reacs):
            Eact[i] = ea[i]
            dH[i] = dh[i]
            rmol.append(Chem.MolFromSmiles(rs[i], smiles_params))
            pmol.append(Chem.MolFromSmiles(ps[i], smiles_params))
        for i in range(half_reacs):
            Eact[i+half_reacs] = ea[i] - dh[i]
            dH[i+half_reacs] = -dh[i]
            rmol.append(Chem.MolFromSmiles(ps[i], smiles_params))
            pmol.append(Chem.MolFromSmiles(rs[i], smiles_params))

    return Eact, dH, rmol, pmol


def split_data(split_method: str, X: ArrayLike, y: ArrayLike, split_ratio:Tuple[float, float], seed: int,
               n_splits: int = 5, index_path=None, verbose=False):
    '''Split data into train/test sets.
    
    Implements methods to split data through Scikit-learn's
    RepeatedKFold cross-validation, train_test_split, 
    ShuffleSplit or manualSplit methods.

    Arguments:
        split_methods: One of ['cv','train_test_split','ShuffleSplit','manualSplit'].
        X: array of x data.
        y: Array of y data.
        split_ratio: Ratio to split data into testing/training by.
        seed: Random number generator seed.
        n_splits: (Optional; Default = 5) Number of shuffle splits or CV folds.
        index_path: (Optional) Path to save split indices to.
        verbose: (Optional; Default: False) Whether or not to pint extra information.
    '''
    if split_method == 'cv':
        kf = RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=seed)
        for train_idx, test_idx in kf.split(X):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
    elif split_method == 'train_test_split':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio[1], random_state=seed,
                                                            shuffle=True)
    elif split_method == 'ShuffleSplit':
        rs = ShuffleSplit(n_splits=n_splits, test_size=split_ratio[1], train_size=split_ratio[0], 
                          random_state=seed)
        for train_idx, test_idx in rs.split(X):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
    elif split_method == 'manualShuffle':
        raise NotImplementedError()

    if index_path is not None:
        if split_method == 'train_test_split':
            if verbose: print('Test/train indices were requested but these are not available for the chosen splitting method.')
        else:
            if verbose: print(f'Saving train/test split indices to {index_path}.')
            np.savez(index_path, train_idx=train_idx, test_idx=test_idx)
    
    return X_train, X_test, y_train, y_test



def normalise(x: ArrayLike, x_mean: float, x_range: float, norm_type: str):
    '''Normalise data.
    
    Data can be mean-normalised (Zscore), shifted, or
    returned without any normalisation.
    
    Arguments:
        x: Array of data to be normalised.
        x_mean: Mean of x array.
        x_range: Standard deviation of x array.
        norm_type: One of ['Zscore', 'shift', 'none'].
    '''
    if norm_type == 'Zscore':
        return (x - x_mean) / x_range
    elif norm_type == 'none':
        return x
    elif norm_type == 'shift':
        return x - x_mean


def un_normalise(x_norm: ArrayLike, x_mean: float, x_range: float, norm_type: str):
    '''Un-normalise data.
    
    Input data can be mean-normalised (Zscore), shifted, or
    without any normalisation. Output will reverse whichever
    normalisation process occurred.
    
    Arguments:
        x_norm: Array of data to be un-normalised.
        x_mean: Mean of original un-normalised x array.
        x_range: Standard deviation of original un-normalised x array.
        norm_type: One of ['Zscore', 'shift', 'none'].
    '''
    if norm_type == 'Zscore':
        return x_norm * x_range + x_mean
    elif norm_type == 'none':
        return x_norm
    elif norm_type == 'shift':
        return x_norm + x_mean