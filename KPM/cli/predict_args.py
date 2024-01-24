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
        parser.add_argument('--direction',
                            type=str,
                            choices=['forward', 'backward', 'both'],
                            default='forward',
                            help='Reaction direction(s) to predict activation energies for.')
        parser.add_argument('--uncertainty',
                            type=str,
                            choices=['True', 'False'],
                            default='False',
                            help='Whether to also return the uncertainty in each prediction..'
                            )
        parser.add_argument('--fix_radicals',
                            type=str,
                            choices=['True', 'False'],
                            default='False',
                            help='Whether to use OBCanonicalRadicals to fix and canonicalise OpenBabel\'s radical structure.'
                            )
        parser.add_argument('--suppress_rdlogs',
                            type=str,
                            choices=['True', 'False'],
                            default='False',
                            help='Whether or not to suppress RDKit\'s output in stderr.')

    @staticmethod
    def run(args):
        '''Run prediction of Eact based on provided args.'''

        mp = ModelPredictor(args)

        fps = mp.process_xyzs()
        mp.predict(fps)
