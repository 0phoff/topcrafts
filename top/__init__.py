"""
TOPCraft
  Random bits and bops that are often useful
  All functions and classes are exposed at the toplevel

Submodules:
"""
import os as _os

# Base imports
from .util import *
from ._log import *

# Metadata
_dir = _os.path.dirname(_os.path.realpath(__file__))
modules = {
    'top.logger': get_module_docstring(_dir + '/_log.py'),
    'top.util': get_module_docstring(_dir + '/util/__init__.py'),
    'top.pd': get_module_docstring(_dir + '/pd/__init__.py'),
    'top.torch': get_module_docstring(_dir + '/torch/__init__.py'),
}

# Conditional imports
try:
    import pandas as _pd
    from .pd import *
except ImportError:
    submodules['top.pd'] = '__disabled__'

try:
    import torch as _torch
    from .torch import *
except ImportError:
    submodules['top.torch'] = '__disabled__'

# Set Docstring
_glob = globals()
for _mod, _string in modules.items():
    _glob['__doc__'] += f'  {_mod}\t{_string}\n'
