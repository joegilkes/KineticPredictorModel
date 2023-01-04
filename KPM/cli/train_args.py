from KPM.cli.shared_args import SharedArgs
from KPM.train import ModelTrainer, ModelTester
from KPM.utils.cli_helper import str_to_tuple
from typing import Tuple

class CLICommand(SharedArgs):
    '''Train a new model for Eact prediction.
    
    Trains a set of neural networks of specified parameters on
    a given dataset of reactions. Contains methods for validating
    newly-created models on test data and saving for later use.
    '''

    @staticmethod
    def add_arguments(parser):
        # Required arguments
        parser.add_argument('dataset',
                            type=str,
                            help='Path to the dataset to be trained from (should be a csv file).')
        parser.add_argument('num_reacs',
                            type=int,
                            help='Number of reactions to load in from the dataset.')

        # Optional arguments
        parser.add_argument('--model_out',
                            type=str,
                            default='./KPM.npz',
                            help='Path to save the trained model to (should be a .npz file).')
        parser.add_argument('--train_direction',
                            type=str,
                            choices=['forward', 'backward', 'both'],
                            default='forward',
                            help='Reaction direction to train from.')
        parser.add_argument('--separate_test_dataset',
                            type=str,
                            default=None,
                            help='Path to separate test dataset (separate train dataset must also be provided).')
        parser.add_argument('--separate_train_dataset',
                            type=str,
                            default=None,
                            help='Path to separate train dataset (separate test dataset must also be provided).')
        parser.add_argument('--norm_type',
                            type=str,
                            choices=['Zscore', 'none', 'shift'],
                            default='Zscore',
                            help='Data normalisation type.')
        parser.add_argument('--split_method',
                            type=str,
                            choices=['cv','train_test_split','ShuffleSplit','manualSplit'],
                            default='cv',
                            help='Data splitting method to split train/test set with.')
        parser.add_argument('--split_ratio',
                            type=str_to_tuple, # Internal conversion of string-form tuple into Python tuple.
                            default='(0.8, 0.2)',
                            help='Ratio of dataset to split into train/test data.')
        parser.add_argument('--split_num',
                            type=int,
                            default=5,
                            help='Number of shuffle splits or CV folds to use.')
        parser.add_argument('--split_index_path',
                            type=str,
                            default=None,
                            help='Path to save cross-validation indices to (saved as numpy .npz file). Leaving as None will not save indices.')
        parser.add_argument('--random_seed',
                            type=int,
                            default=None,
                            help='Seed for the random number generator. Leaving as None will initialise from a random seed.')
        parser.add_argument('--training_prediction_dir',
                            type=str,
                            default=None,
                            help='Directory to save training prediction values to (saved as numpy .npz files). Leaving as None will not save predictions.')
        parser.add_argument('--save_test_train_path',
                            type=str,
                            default=None,
                            help='Path to save X_train/test and y_train/test data to (saved as numpy .npz file). Leaving as None will not save data.')
        parser.add_argument('--plot_dir',
                            type=str,
                            default='./',
                            help='Directory to save correlation plots to.')

        # Neural network optional arguments
        # See https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html for full info.
        parser.add_argument('--opt_hyperparams',
                            type=str,
                            choices=['True', 'False'],
                            default='False',
                            help='Whether to perform a grid search for optimal neural network hyperparameters.')
        parser.add_argument('--opt_hyperparams_jobs',
                            type=int,
                            default=None,
                            help='Number of parallel threads to perform hyperparameter optimisation over.')
        parser.add_argument('--opt_hyperparams_file',
                            type=str,
                            default=None,
                            help='Path to JSON file containing hyperparameter lists to optimise over.')
        parser.add_argument('--nn_activation_function',
                            type=str,
                            choices=['identity', 'logistic', 'tanh','relu'],
                            default='relu',
                            help='Neural network activation function to use.')
        parser.add_argument('--nn_ensemble_size',
                            type=int,
                            default=5,
                            help='Number of neural network model runs (ensembles).')
        parser.add_argument('--nn_solver',
                            type=str,
                            choices=['lbfgs', 'sgd', 'adam'],
                            default='adam',
                            help='The solver to use in neural network weight optimisation.')
        parser.add_argument('--nn_layers',
                            type=int,
                            nargs='+',
                            default=100,
                            help='Hidden layer sizes in neural network. Accepts multiple arguments (e.g. \'--nn_layers 100 100\' gives a network with two hidden layers of 100 neaurons each).')
        parser.add_argument('--nn_alpha',
                            type=float,
                            default=1e-4,
                            help='Strength of neural network L2 regularisation.')
        parser.add_argument('--nn_max_iters',
                            type=int,
                            default=500,
                            help='Maximum number of iterations in neural network training.')
        parser.add_argument('--nn_learning_rate',
                            type=str,
                            choices=['constant', 'invscaling', 'adaptive'],
                            default='constant',
                            help='The learning rate schedule for neural network weight updates.')
        parser.add_argument('--nn_learning_rate_init',
                            type=float,
                            default=1e-3,
                            help='The initial neural network learning rate.')
        parser.add_argument('--nn_learning_rate_max',
                            type=float,
                            default=1e-2,
                            help='The maximum neural network learning rate.')

        # Featurisation optional arguments
        parser.add_argument('--descriptor_type',
                            type=str,
                            choices=['soap', 'mbtr', 'MorganF'],
                            default='MorganF',
                            help='The type of molecular descriptor to train on.')
        parser.add_argument('--similarity_type',
                            type=str,
                            choices=['tanimoto', 'fraggle'],
                            default='fraggle',
                            help='The type of similarity score to use in training.')
        parser.add_argument('--smiles_type',
                            type=str,
                            choices=['obabel', 'canonical'],
                            default='canonical',
                            help='The type of smiles descriptor to use in training.')
        parser.add_argument('--morgan_num_bits',
                            type=int,
                            default=1024,
                            help='Morgan fingerprint size (bits).')
        parser.add_argument('--morgan_radius',
                            type=int,
                            default=5,
                            help='Morgan fingerprint radius.')


    @staticmethod
    def run(args):
        '''Train neural networks given the supplied arguments.'''

        # Train model and save.
        trainer = ModelTrainer(args)
        X_train, X_test, y_train, y_test = trainer.process()
        trainer.run(X_train, y_train)

        # Test model on both training and testing data.
        tester = ModelTester(args.model_out)

        Eact_pred_train, Eact_uncert_train = tester.predict(X_train, y_train, 'train')
        tester.plot_correlation(y_train, Eact_pred_train, Eact_uncert_train, 'train')

        Eact_pred_test, Eact_uncert_test = tester.predict(X_test, y_test, 'test')
        tester.plot_correlation(y_test, Eact_pred_test, Eact_uncert_test, 'test')

        print('KPM finished.')
        input('Press ENTER to close.')