class SharedArgs:
    '''Base class for adding shared arguments to parser.
    
    Shared arguments are arguments that are used by both
    the train and predict functions of KPM.
    '''

    @staticmethod
    def add_shared_arguments(parser):
        '''Adds arguments to parser that get shared by all subparsers.'''
        parser.add_argument('--verbose',
                            type=str,
                            choices=['True', 'False'],
                            default='False',
                            help='Whether or not to suppress extra print statements.')