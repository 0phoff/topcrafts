"""Everything related to reading/writing matlab stuff with pandas."""
from collections import Iterable
import pandas as pd
import scipy.io as io

__all__ = ['loadmat']

def loadmat(filename, variables=None, data_slice=None, columns=None, **kwargs):
    """ This function loads a matlab .mat file and creates a pd.DataFrame from all the variables given.

    TODO:
        explain arguments
    """
    raw_data = io.loadmat(filename, **kwargs)

    # preprocess parameters
    if variables is None:
        keys = raw_data.keys()
        
        variables = []
        for k in keys:
            if not k.startswith('__'):
                variables.append(k)
    elif not isinstance(variables, Iterable):
        variables = [variables]
    if data_slice is None:
        data_slice = [slice(None)] * len(variables)
    elif not isinstance(data_slice, Iterable):
        data_slice = [data_slice] * len(variables)
    if columns is None:
        columns = [None] * len(variables)
    elif not isinstance(columns, Iterable) or not isinstance(columns[0], (Iterable, slice)):
        columns = [columns] * len(variables)

    # check parameters
    if len(variables) != len(data_slice) != len(columns):
        raise TypeError('Each variable should have its own data_slice and columns or a single data_slice and column for all variables')

    output = dict()
    for i, var in enumerate(variables):
        if isinstance(columns[i], slice):
            out = pd.DataFrame(raw_data[var][data_slice[i]], columns=raw_data[var][columns[i]][0])
        else:
            out = pd.DataFrame(raw_data[var][data_slice[i]], columns=columns[i])

        output[var] = out

    if len(output) == 1:
        (_, v), = output.items()
        return v
    else:
        return output
