from KPM.cli.shared_args import SharedArgs
from KPM.predict import ModelPredictor

class CLICommand(SharedArgs):
    '''Use a previously trained KPM model to predict Eact for a reaction.
    
    Takes in a model trained by KPM, along with xyz files for
    the reactant and product structures, and the enthalpy between
    these two structures.
    
    Returns a prediction of the activation energy (Eact) for the
    reaction defined by these structures.
    '''

    @staticmethod
    def add_arguments(parser):
        # Required arguments
        parser.add_argument('model',
                            type=str,
                            help='Path to KPM-trained prediction model (as pickle file).')
        parser.add_argument('reactant',
                            type=str,
                            help='Path to reactant xyz structure file (may contain multiple structures).')
        parser.add_argument('product',
                            type=str,
                            help='Path to product xyz structure file (may contain multiple structures).')
        parser.add_argument('enthalpy',
                            type=str,
                            help='Path to text file containing enthalpy/enthalpies of reaction(s) defined by input reactants and products.')

        # Optional arguments
        parser.add_argument('--outfile',
                            type=str,
                            default='./KPM_out.txt',
                            help='Path to prediction output text file.')

    @staticmethod
    def run(args):
        '''Run prediction of Eact based on provided args.'''

        mp = ModelPredictor(args)

        diff = mp.process_xyzs()
        mp.predict(diff)
