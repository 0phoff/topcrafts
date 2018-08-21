"""Random utilitary stuff for profiling"""
import gc
import logging

__all__ = ['ToggleGC']
log = logging.getLogger(__name__)


class ToggleGC:
    """ Context manager to disable garbage collection.

    Args:
        flag (boolean): Whether to enable the garbage collector or not

    Example:
        >>> import gc
        >>> prev = gc.isenabled()
        >>> with ToggleGC(False):
        >>>     assert not gc.isenabled()
        >>>     with ToggleGC(True):
        >>>         assert gc.isenabled()
        >>>     assert not gc.isenabled()
        >>> assert gc.isenabled() == prev
    """
    def __init__(self, flag):
        self.flag = flag
        self.prev = None

    def __enter__(self):
        self.prev = gc.isenabled()
        if self.flag:
            gc.enable()
            log.debug('Enabled Garbage Collection')
        else:
            gc.disable()
            log.debug('Disabled Garbage Collection')

    def __exit__(self, ex_type, ex_value, trace):
        if self.prev:
            gc.enable()
            log.debug('Enabled Garbage Collection')
        else:
            gc.disable()
            log.debug('Disabled Garbage Collection')
