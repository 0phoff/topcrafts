"""
TOP Crafts: Random bits and bops that are often useful
"""
import os as _os


# Base imports
from .util import *

# Metadata
_dir = _os.path.dirname(_os.path.realpath(__file__))
submodules = {
    'top.util': get_module_docstring(_dir + '/util/__init__.py'),
    'top.pd': get_module_docstring(_dir + '/pd/__init__.py'),
}

# Conditional imports
try:
    import pandas as _pd
    from .pd import *
except ImportError:
    submodules['top.pd'] = '__disabled__'
