import numpy as np


def printf(s, e=', '):
    """
    Alternative print function to print without new line. Allow more compact code.
    e: print termination. A comma here, against a new line for the default print.
    """
    print(s, end=e)
    return None


def print_params(algo, exp, **kwargs):
    """
    Print all params used during the run of an experiment.
    Allows to check easily all the chosen parameters.
    algo: Name of the run algorithm (string, eg 'Sinkhorn' or 'Descent')
    exp: id of the experiment (positive integer)
    """
    printf('Train {} expe {}'.format(algo, exp), e=': ')
    keys = list(kwargs.keys())
    for u in keys[:-1]:
        printf(u+'={}'.format(kwargs[u]))
    print(keys[-1]+'={}.'.format(kwargs[keys[-1]]))
    return None
