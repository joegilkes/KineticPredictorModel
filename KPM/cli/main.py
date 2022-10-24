'''Command line interface for KPM, heavily inspired by ASE's interface 
(https://gitlab.com/ase/ase/-/blob/master/ase/cli/main.py).
'''

import argparse
import textwrap
from importlib import import_module

# Needed here or something else imports it with a different backend.
import matplotlib
import os
if 'DISPLAY' in os.environ.keys():
    if 'DISPLAY' == '':
        _mpl_backend = 'Agg'
    else:
        _mpl_backend = 'TkAgg'
else:
    _mpl_backend = 'Agg'
print(_mpl_backend)
matplotlib.use(_mpl_backend)


class CLIError(Exception):
    '''Error for CLI commands.

    A subcommand may raise this.  The message will be forwarded to
    the error() method of the argument parser.
    '''


commands = [
    ('train', 'KPM.cli.train_args'),
    ('test', 'KPM.cli.test_args'),
    ('predict', 'KPM.cli.predict_args')
]


def main(prog='KPM', description='KPM command line tool',
         commands=commands, args=None):
    '''Connect main parser to relevant subparser and execute KPM.'''
    parser = argparse.ArgumentParser(prog=prog,
                                     description=description,
                                     formatter_class=Formatter)
    subparsers = parser.add_subparsers(title='Sub-commands',
                                       dest='command')
    subparser = subparsers.add_parser('help',
                                      description='Help',
                                      help='Help for sub-command.')
    subparser.add_argument('helpcommand',
                           nargs='?',
                           metavar='sub-command',
                           help='Provide help for sub-command.')

    functions = {}
    parsers = {}
    for command, module_name in commands:
        cmd = import_module(module_name).CLICommand
        docstring = cmd.__doc__
        parts = docstring.split('\n', 1)
        if len(parts) == 1:
            short = docstring
            long = docstring
        else:
            short, body = parts
            long = short + '\n' + textwrap.dedent(body)
        subparser = subparsers.add_parser(
            command,
            formatter_class=Formatter,
            help=short,
            description=long)
        cmd.add_arguments(subparser)
        cmd.add_shared_arguments(subparser)
        functions[command] = cmd.run
        parsers[command] = subparser

    args = parser.parse_args(args)

    if args.command == 'help':
        if args.helpcommand is None:
            parser.print_help()
        else:
            parsers[args.helpcommand].print_help()
    elif args.command is None:
        parser.print_usage()
    else:
        f = functions[args.command]
        try:
            if f.__code__.co_argcount == 1:
                f(args)
            else:
                f(args, parsers[args.command])
        except KeyboardInterrupt:
            pass
        except CLIError as x:
            parser.error(x)
        except Exception as x:
            raise x


class Formatter(argparse.ArgumentDefaultsHelpFormatter):
    '''Improved help formatter.'''

    def _fill_text(self, text, width, indent):
        assert indent == ''
        out = ''
        blocks = text.split('\n\n')
        for block in blocks:
            if block[0] == '*':
                # List items:
                for item in block[2:].split('\n* '):
                    out += textwrap.fill(item,
                                         width=width - 2,
                                         initial_indent='* ',
                                         subsequent_indent='  ') + '\n'
            elif block[0] == ' ':
                # Indented literal block:
                out += block + '\n'
            else:
                # Block of text:
                out += textwrap.fill(block, width=width) + '\n'
            out += '\n'
        return out[:-1]
