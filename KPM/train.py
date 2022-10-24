'''KPM Neural Network Training Module

Used to create new KPM models and test their accuracy on
given molecular datasets.
'''

from KPM.utils.data_funcs import load_dataset, extract_data, split_data
from KPM.utils.data_funcs import normalise, un_normalise
from KPM.utils.descriptors import calc_diffs

from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import os

from numpy.typing import ArrayLike

class ModelTrainer:
    '''Trains a neural network on a dataset of reactions.
    
    Takes in all required arguments through train_args.
    Datasets should be provided as csv files in the same
    format as the supplied b97d3 dataset.

    Arguments:
        args: argparse Namespace from CLI.
    '''
    def __init__(self, args):
        '''Initialise class from supplied CLI arguments.'''

        print('--------------------------------------------')
        print('KPM Model Training')
        print('--------------------------------------------\n')

        self.argparse_args = args

        self.dataset = args.dataset
        self.num_reacs = args.num_reacs

        self.model_out = args.model_out
        self.train_direction = args.train_direction
        self.separate_test_dataset = args.separate_test_dataset
        self.separate_train_dataset = args.separate_train_dataset
        self.norm_type = args.norm_type
        self.split_method = args.split_method
        self.split_ratio = args.split_ratio
        self.split_num = args.split_num
        self.split_index_path = args.split_index_path
        self.random_seed = args.random_seed
        self.training_prediction_dir = args.training_prediction_dir
        self.save_test_train_path = args.save_test_train_path
        self.plot_dir = args.plot_dir
        self.nn_activation_function = args.nn_activation_function
        self.nn_ensemble_size = args.nn_ensemble_size
        self.nn_solver = args.nn_solver
        self.nn_learning_rate = args.nn_learning_rate
        self.nn_learning_rate_init = args.nn_learning_rate_init
        self.nn_learning_rate_max = args.nn_learning_rate_max # Not used yet, doesn't look like this is an option.
        self.descriptor_type = args.descriptor_type # Only MorganF currently implemented.
        self.similarity_type = args.similarity_type # Not used yet.
        self.smiles_type = args.smiles_type # Not used yet.
        self.morgan_num_bits = args.morgan_num_bits
        self.morgan_radius = args.morgan_radius
        self.verbose = True if args.verbose == 'True' else False

        # Sanitise input for separate test/train datasets.
        if self.separate_train_dataset is not None:
            if self.separate_test_dataset is None:
                raise ValueError('If separate_train_dataset is specified, separate_test_dataset must also be specified.')
        elif self.separate_test_dataset is not None:
            if self.separate_train_dataset is None:
                raise ValueError('If separate_test_dataset is specified, separate_train_dataset must also be specified.')

        # Modify number of reactions based on train_direction.
        if self.train_direction == 'both':
            self.num_reacs = 2*self.num_reacs


    def process(self):
        '''Process the data into training and test datasets.'''
        # If loading a single combined dataset that needs to be split...
        if self.separate_test_dataset is None:
            print('Taking train/test data from a single dataset.')
            ea, dh, rs, ps = load_dataset(self.dataset)

            # Extract and transform data dependent on train_direction.
            Eact, dH, rmol, pmol = extract_data(ea, dh, rs, ps, self.num_reacs, self.train_direction)

            avg_Eact = np.mean(Eact)
            avg_dH = np.mean(dH)
            if self.verbose:
                print(f'\nLength of Eact = {len(Eact)}, Mean Eact = {avg_Eact} kcal/mol')
                print(f'Length of dH = {len(dH)}, Mean dH = {avg_dH} kcal/mol')
                print(f'Total number of MOL objects = {len(rmol)}\n')

            # Normalise Eact.
            std_Eact = np.std(Eact)
            Eact = normalise(Eact, avg_Eact, std_Eact, self.norm_type)

            if self.verbose: print('Data sorted. Calculating reaction difference fingerprints.')

            # Calculate reaction difference fingerprints.
            diffs = calc_diffs(self.num_reacs, self.descriptor_type, rmol, pmol, dH, 
                               self.morgan_radius, self.morgan_num_bits)

            print('Fingerprint calculation complete.')
            if self.verbose: print(f'\nSplitting train/test data by {self.split_method}.')

            # Split data into train/test sets.
            X_train, X_test, y_train, y_test = split_data(self.split_method, diffs, Eact, self.split_ratio,
                                                          self.random_seed, self.split_num, self.split_index_path,
                                                          self.verbose)
        # Else if loading separate train/test datasets...
        else:
            print('Taking train/test data from separate datasets.')
            ea_train, dh_train, rs_train, ps_train = load_dataset(self.separate_train_dataset)
            ea_test, dh_test, rs_test, ps_test = load_dataset(self.separate_test_dataset)

            # Extract and transform data dependent on train_direction.
            num_train_reacs = len(ea_train)
            num_test_reacs = len(ea_test)
            Eact_train, dH_train, rmol_train, pmol_train = extract_data(ea_train, dh_train, rs_train, ps_train, 
                                                                        num_train_reacs, self.train_direction)
            Eact_test, dH_test, rmol_test, pmol_test = extract_data(ea_test, dh_test, rs_test, ps_test, 
                                                                    num_test_reacs, self.train_direction)

            avg_Eact = np.mean(Eact_train)
            avg_dH_train = np.mean(dH_train)
            avg_Eact_test = np.mean(Eact_test)
            avg_dH_test = np.mean(dH_test)
            if self.verbose:
                print(f'Length of Eact (train) = {len(Eact_train)}, Mean Eact (train) = {avg_Eact} kcal/mol')
                print(f'Length of dH (train) = {len(dH_train)}, Mean dH (train) = {avg_dH_train} kcal/mol')
                print(f'Total number of training MOL objects = {len(rmol_train)}')
                print(f'Length of Eact (test) = {len(Eact_test)}, Mean Eact (test) = {avg_Eact_test} kcal/mol')
                print(f'Length of dH (test) = {len(dH_test)}, Mean dH (test) = {avg_dH_test} kcal/mol')
                print(f'Total number of testing MOL objects = {len(rmol_test)}')

            # Normalise Eact from training data.
            std_Eact = np.std(Eact_train)
            Eact_train = normalise(Eact_train, avg_Eact, std_Eact, self.norm_type)
            Eact_test = normalise(Eact_test, avg_Eact, std_Eact, self.norm_type)

            if self.verbose: print('Data sorted. Calculating reaction difference fingerprints.')

            # Calculate reaction difference fingerprints.
            diffs_train = calc_diffs(num_train_reacs, self.descriptor_type, rmol_train, pmol_train, dH_train,
                                     self.morgan_radius, self.morgan_num_bits)
            diffs_test = calc_diffs(num_test_reacs, self.descriptor_type, rmol_test, pmol_test, dH_test,
                                     self.morgan_radius, self.morgan_num_bits)

            print('Fingerprint calculation complete.')

            X_train = diffs_train
            X_test = diffs_test
            y_train = Eact_train
            y_test = Eact_test

        if self.save_test_train_path is not None:
            np.savez(self.save_test_train_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
            if self.verbose: print(f'Saved train/test data to {self.save_test_train_path}.\n')

        print('Train/test data ready.')

        self.norm_avg_Eact = avg_Eact
        self.norm_std_Eact = std_Eact
        return X_train, X_test, y_train, y_test


    def run(self, X_train: ArrayLike, y_train: ArrayLike):
        '''Runs the model training procedure.
        
        Arguments:
            X_train: Array of reaction difference fingerprints.
            y_train: Array of normalised activation energies.
        '''
        print(f'Training {self.nn_ensemble_size} neural networks.')
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        if self.verbose: print('Scaled data.')

        regr = []
        for i in range(self.nn_ensemble_size):
            if self.verbose: print(f'Training NN {i+1}...')
            regr.append(MLPRegressor(hidden_layer_sizes=(200), activation=self.nn_activation_function,
                                     solver=self.nn_solver, learning_rate=self.nn_learning_rate,
                                     learning_rate_init=self.nn_learning_rate_init, max_iter=1000,
                                     random_state=self.random_seed
                                     ).fit(X_train_scaled, y_train))

        if self.verbose: print(f'Saving models to {self.model_out}')
        print('Training complete!\n')
        
        norm_vars = [self.norm_avg_Eact, self.norm_std_Eact]
        np.savez(self.model_out, models=regr, scaler=scaler, norm_vars=norm_vars, args=self.argparse_args)


class ModelTester:
    '''Tests prediction accuracy of a KPM model.
    
    Loads in a previously trained model, extracting arguments
    directly from it. Uses a secondary dataset as test data,
    and predicts Eact values for this dataset, producing a
    correlation plot for these predictions.

    Arguments:
        model_path: Path to pickle file containing KPM model(s).
        test_args: argparse Namespace from running KPM in 'test' mode.
    '''
    def __init__(self, model_path: str, test_dataset: str = None, test_num_reacs: int = None,
                 plot_dir: str = './', verbose: str = 'False'):
        '''Initialise KPM model tester.'''

        print('--------------------------------------------')
        print('KPM Model Testing')
        self.verbose = True if verbose == 'True' else False

        model = np.load(model_path, allow_pickle=True)
        self.regr = model['models']
        self.scaler = model['scaler'][()] # scaler gets saved as a 0D array.
        norm_vars = model['norm_vars']
        self.norm_avg_Eact = norm_vars[0]
        self.norm_std_Eact = norm_vars[1]
        self.training_args = model['args'][()] # args gets saved as a 0D array.
        args = self.training_args

        self.orig_dataset = args.dataset
        self.orig_num_reacs = args.num_reacs
        self.model_out = args.model_out
        self.train_direction = args.train_direction
        self.separate_test_dataset = args.separate_test_dataset
        self.separate_train_dataset = args.separate_train_dataset
        self.norm_type = args.norm_type
        self.split_method = args.split_method
        self.split_ratio = args.split_ratio
        self.split_num = args.split_num
        self.split_index_path = args.split_index_path
        self.random_seed = args.random_seed
        self.training_prediction_dir = args.training_prediction_dir
        self.save_test_train_path = args.save_test_train_path
        self.plot_dir = args.plot_dir
        self.nn_activation_function = args.nn_activation_function
        self.nn_ensemble_size = args.nn_ensemble_size
        self.nn_solver = args.nn_solver
        self.nn_learning_rate = args.nn_learning_rate
        self.nn_learning_rate_init = args.nn_learning_rate_init
        self.nn_learning_rate_max = args.nn_learning_rate_max # Not used yet, doesn't look like this is an option.
        self.descriptor_type = args.descriptor_type
        self.similarity_type = args.similarity_type # Not used yet.
        self.smiles_type = args.smiles_type # Not used yet.
        self.morgan_num_bits = args.morgan_num_bits
        self.morgan_radius = args.morgan_radius

        # 
        if test_dataset is not None:
            print(f'Testing with data from {test_dataset}')
            self.dataset = test_dataset
            self.num_reacs = test_num_reacs
            self.plot_dir = plot_dir
            # Modify number of reactions based on train_direction.
            if self.train_direction == 'both':
                self.num_reacs = 2*self.num_reacs
        else:
            print('Testing with data from original test/train split.')

        print('--------------------------------------------\n')


    def process_test_data(self):
        '''Process a new dataset into X_test and y_test arrays.
        
        When in test-only mode and provided with a new dataset of
        reactions, runs through the process of extracting data
        and creating a test set to run predictions on.
        '''
        if self.dataset is None:
            raise RuntimeError('This function can only be run when new test data is provided!')

        print('Loading in new test dataset.\n')
        # print(self.dataset, type(self.dataset))
        ea, dh, rs, ps = load_dataset(self.dataset)

        # Extract and transform data dependent on train_direction.
        Eact, dH, rmol, pmol = extract_data(ea, dh, rs, ps, self.num_reacs, self.train_direction)

        avg_Eact = np.mean(Eact)
        avg_dH = np.mean(dH)
        if self.verbose:
            print(f'\nLength of Eact = {len(Eact)}, Mean Eact = {avg_Eact} kcal/mol')
            print(f'Length of dH = {len(dH)}, Mean dH = {avg_dH} kcal/mol')
            print(f'Total number of MOL objects = {len(rmol)}\n')

        # Normalise Eact
        Eact = normalise(Eact, self.norm_avg_Eact, self.norm_std_Eact, self.norm_type)

        if self.verbose: print('Data loaded. Calculating reaction difference fingerprints.')

        # Calculate reaction difference fingerprints.
        diffs = calc_diffs(self.num_reacs, self.descriptor_type, rmol, pmol, dH, self.morgan_radius, self.morgan_num_bits)
        print('Fingerprint calculation complete.')

        return diffs, Eact


    def predict(self, X: ArrayLike, y: ArrayLike, data_type):
        '''Use the loaded model to predict Eact for a train/test dataset.
        
        Generic function to predict Eact values for either training or
        test data across an ensemble of neural networks. Uses actual Eact
        values for the given reactions to return R**2 values for each
        model in the ensemble.
        
        Arguments:
            X: Array of reaction difference fingerprints.
            y: Actual Eact values for these reactions.
            data_type: Either 'train' or 'test', determines verbose output.
        '''
        if data_type not in ['train', 'test']:
            raise ValueError('Unknown data_type! Must be either \'train\' or \'test\'.')

        print(f'Predicting Eact values for {data_type}ing data.')
        if self.verbose:
            print(f'Predicting Eact across {self.nn_ensemble_size} NNs.\n')

        n_reacs = len(X)
        X = self.scaler.transform(X)

        Eact_pred = np.zeros(n_reacs)
        r2s = np.zeros(self.nn_ensemble_size)
        for i in range(self.nn_ensemble_size):
            pred = self.regr[i].predict(X)
            pred = un_normalise(pred, self.norm_avg_Eact, self.norm_std_Eact, self.norm_type)
            Eact_pred += pred
            r2 = self.regr[i].score(X, y)
            r2s[i] = r2
            if self.verbose: print(f'NN{i} R2 = {r2}')

        Eact_pred = Eact_pred / self.nn_ensemble_size
        if self.verbose: print(f'Average R2 = {np.mean(r2s)}')

        # Save predictions if requested.
        if self.training_prediction_dir is not None:
            save_path = os.path.join(self.training_prediction_dir, f'Eact_pred_{data_type}.npz')
            np.savez(save_path, Eact_pred=Eact_pred)
            if self.verbose: print(f'Predictions saved to {save_path}\n')

        return Eact_pred

    
    def plot_correlation(self, y_true: ArrayLike, y_pred: ArrayLike, data_type: str):
        '''Use the loaded model to plot correlation between predicted and actual Eact values.
        
        Generic function to plot correlation plots for either training or
        test data. Prints a variety of error metrics for given data.

        Arguments:
            y_true: Actual Eact values for reactions in dataset.
            y_pred: Predicted Eact values for reactions in dataset.
            data_type: Either 'train' or 'test', determines plot wording and colours.
        '''
        if data_type not in ['train', 'test']:
            raise ValueError('Unknown data_type! Must be either \'train\' or \'test\'.')

        print(f'Analysis on {data_type}ing data prediction:')

        # Un-normalise true data.
        y_true = un_normalise(y_true, self.norm_avg_Eact, self.norm_std_Eact, self.norm_type)
        
        # Calculate error metrics.
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        print(f'Mean Absolute Error (MAE) in {data_type}ing data prediction:      {mae} kcal/mol')
        print(f'Mean Squared Error (MSE) in {data_type}ing data prediction:       {mse} kcal/mol')
        print(f'Root Mean Squared Error (RMSE) in {data_type}ing data prediction: {rmse} kcal/mol\n')

        # Plot correlation.
        if self.verbose: print('Plotting correlation between true and predicted values.')
        col = 'blue' if data_type == 'train' else 'purple'
        fignum = 1 if data_type == 'train' else 2

        fig = plt.figure(fignum, figsize=(8, 6), dpi=100)
        plt.plot(y_true, y_true, color='orange', lw=3)
        plt.plot(y_pred, y_true, 'o', color=col, mfc='white', markersize=8, mew=2, alpha=0.5)
        plt.xlabel(r"Predicted E$_a$ (kcal/mol)",fontsize=16)
        plt.ylabel(r"True E$_a$ (kcal/mol)",fontsize=16)
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        if self.plot_dir is not None:
            save_path = os.path.join(self.plot_dir, f'corr_{data_type}.png')
            if not os.path.exists(self.plot_dir): os.mkdir(self.plot_dir)
            plt.savefig(save_path, dpi=300)
            if self.verbose: print(f'Saved plot to {save_path}\n')
        
        plt.pause(0.01)