from KPM.cli.shared_args import SharedArgs
from KPM.train import ModelTester

class CLICommand(SharedArgs):
    '''Test a previously trained KPM model on a specified dataset.
    
    Tests prediction accuracy of a previously trained KPM model
    on a separate test dataset, returning a correlation plot for
    this data.

    Extracts arguments from saved model to ensure internal consistency
    in predictions, so only requires a relatively sparse command line
    interface.
    '''

    @staticmethod
    def add_arguments(parser):
        # Required arguments
        parser.add_argument('model',
                            type=str,
                            help='Path to KPM-trained prediction model (as npz file).')
        parser.add_argument('dataset',
                            type=str,
                            help='Path to the dataset to load test data from (should be a csv file).')
        parser.add_argument('num_reacs',
                            type=int,
                            help='Number of reactions to load in from the dataset.')

        # Optional arguments
        parser.add_argument('--plot_dir',
                            type=str,
                            default='./',
                            help='Directory to save correlation plots to.')

    @staticmethod
    def run(args):
        '''Run predictions on the supplied model with the supplied test dataset.'''

        # Test model on new training dataset.
        tester = ModelTester(args.model, args.dataset, args.num_reacs, args.plot_dir, args.verbose)

        X_test, y_test = tester.process_test_data()
        Eact_pred_test = tester.predict(X_test, y_test, 'test')
        tester.plot_correlation(y_test, Eact_pred_test, 'test')

        print('KPM finished.')
        input('Press ENTER to close.')
        