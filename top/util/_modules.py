"""Functions related to handling python (sub)modules."""
import code

__all__ = ['get_module_docstring']


def get_module_docstring(filepath):
    """Get module-level docstring of Python module at filepath.
    
    Args:
        filepath(string or Path): Path to the file
    """
    co = compile(open(filepath).read(), filepath, 'exec')
    if co.co_consts and isinstance(co.co_consts[0], str):
        docstring = co.co_consts[0]
    else:
        docstring = None
    return docstring
