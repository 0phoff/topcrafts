"""Time your code execution"""
import time
import logging
from functools import wraps
from tqdm import tqdm
import numpy as np
from ._util import ToggleGC

__all__ = ['Timer', 'Timeit']
log = logging.getLogger(__name__)


class Timer:
    """This class is your one stop shop for basic timing functionality.
    It can be used with the start/time/stop methods, as a context manager or as a decorator. |br|
    Code adapted from timerit_ module.

    Args:
        label (str, optional): Label for the time logger; Default **top.timer**
        verbose (Boolean, optional): Whether to log the time; Default **True**
        format (str, optional): Formatting for logging the time; Default **''**
        unit (str, optional): unit of time; choices: s, ms, us, ns; Default **s**

    Attributes:
        self.value: Timed value after running stop(), context manager or decorated function; Default **None**

    Example:
        >>> t = top.Timer(unit='ns')
        >>> t.start()
        >>> a = np.arange(12).reshape(3,4)
        >>> t1 = t.time()
        >>> b = a.sum()
        >>> t2 = t.time()
        >>> t.stop()
        >>> t3 = t.value

        >>> t = top.Timer(unit='ms', verbose=False)
        >>> with t:
        ...     a = np.arange(12).reshape(3,4)
        ...     b = a.sum()
        >>> print(t.value, t.unit)

        >>> @top.Timer(unit='us', format='.3f')
        ... def custom_sum_func(arr):
        ...     return arr.sum()
        >>> a = np.arange(12).reshape(3,4)
        >>> b = custom_sum_func(a)

    _timerit: https://www.github.com/Erotemic/timerit
    """
    _time = time.process_time
    _units = {
        's': 1e0,
        'ms': 1e3,
        'us': 1e6,
        'ns': 1e9,
    }

    def __init__(self, label='top.timer', verbose=True, format='', unit='s'):
        self.verbose = verbose
        self.format = format
        self.unit = unit
        self.log = logging.getLogger(label)
        
        if self.unit not in self._units:
            log.error(f'{self.unit} is not a valid unit, setting to default of "s" [{list(self._units.keys())}]')
            self.unit = 's'

        self.start()

    @property
    def label(self):
        """ Change the time logger label """
        return self.log.name

    @label.setter
    def label(self, value):
        self.log.name = value

    def start(self):
        """ Set the start time """
        self.value = None
        self._start = self._time()

    def stop(self, msg=''):
        """ Set timer.value to the elapsed time and log it if verbose is True """
        t2 = self._time()
        self.value = t2 - self._start
        self.value *= self._units[self.unit]

        if self.verbose:
            if len(msg) != 0:
                msg += ': '
            self.log.profile(f'{msg}{self.value:{self.format}} {self.unit}')

        self._start += self._time() - t2

    def time(self, msg=''):
        """ Return the elapsed time and log it if verbose is True """
        t2 = self._time()
        time = t2 - self._start
        time *= self._units[self.unit]

        if self.verbose:
            if len(msg) != 0:
                msg += ': '
            self.log.profile(f'{msg}{time:{self.format}} {self.unit}')

        self._start += self._time() - t2
        return time

    def __call__(self, fn):
        """ Wrap a function with this class as a decorator.
        This will log the time the function took and use the function name as label.
        """
        @wraps(fn)
        def time_wrapper(*args, **kwargs):
            verbose = self.verbose
            self.verbose = True
            label = self.label
            self.label = fn.__name__
            with self:
                ret = fn(*args, **kwargs)

            self.label = label
            self.verbose = verbose
            return ret

        return time_wrapper

    def __enter__(self):
        """ Use this class as a context manager.
        The elapsed time will be stored as timer.value in the end.
        You can also call the time() function to get intermediate results.
        """
        self._start = self._time()
        return self

    def __exit__(self, ex_type, ex_value, trace):
        self.value = self._time() - self._start
        self.value *= self._units[self.unit]

        if self.verbose:
            self.log.profile(f'Elapsed Time: {self.value:{self.format}} {self.unit}')

        if trace is not None:
            return False


class Timeit:
    """ Execute a code block multiple times and collect execution times.
    This class is like the standard library's :func:`timeit.timeit` function, but has a nicer interface. |br|
    This class yields a :class:`top.prof.Timer` instance that can be used to time the execution you need.
    If you dont use the returned Timer instance, a global timer will be used that times the entire loop. |br|
    Code adapted from timerit_ module.
    
    Args:
        num (int, optional): Number of times to execute the timed code block; Default **1**
        label (str, optional): Label for the profile logger and tqdm barl Default **None**
        verbose (boolean or int, optional): Verbosity level (see Note); Default **2**
        tqdm (boolean, optional): Whether to use tqdm; Default **False**
        format (str, optional): Formatting for logging the time; Default **''**
        unit (str, optional): unit of time; choices: s, ms, us, ns; Default **s**
        gc (boolean, optional): Whether to enable the garbage collector during timing; Default **False**

    Attributes:
        self.values: This list contains the different timed values after execution of timeit; Default **[]**

    Warning:
        To get more accurate results, it is highly recommended to use the yielded timer from this class
        instead of the global one.

    Note:
        This class allows for multiple verbosity levels:
            - _0_  Nothing gets logged
            - _1_  Log the best time after the timing loop finishes
            - _2_  Log the best and mean±std time after the timing loop finishes
            - _3_  If tqdm is False, log each time during the loop. Also log the best and mean±std after the timing loop finishes

        You can also use **True** and **False** which correspond to the levels _1_ and _0_ respectively.

    Note:
        Just like the regular python timeit function, the garbage collector gets disabled during the time loop by default.
        This allows for more consistent timing results, but it also means the timed values might differ slightly from real use cases.

    Example:
        >>> # Using global timer
        >>> for _ in top.Timeit(1000):
        ...     a = np.random.rand(1000, 100)
        ...     b = a.mean(axis=1).sum()

        >>> # Using timer instance
        >>> for t in top.Timeit(1000, tqdm=True, unit='us', format='.3f'):
        ...     a = np.random.rand(1000, 100)   # Setup code
        ...     with t:                         # Timed code
        ...         b = a.mean(axis=1).sum()

        >>> # Dont print anything but use times list
        >>> timeit = top.Timeit(1000, verbose=False)
        >>> for timer in timeit:
        ...     a = np.random.rand(1000, 100)
        ...     timer.start()
        ...     b = a.mean(axis=1).sum()
        ...     timer.stop()
        >>> min_time = min(timeit.values)
        >>> timeit.reset()  # Ready for new run

    _timerit: https://www.github.com/Erotemic/timerit
    """
    def __init__(self, num=1, label=None, verbose=2, tqdm=False, format='', unit='s', gc=False):
        self.num = num
        self.label = label
        self.verbose = verbose
        self.tqdm = tqdm
        self.format = format
        self.unit = unit
        self.gc = gc
        self.log = logging.getLogger(label)
        
        self.reset()

    def reset(self):
        self.values = []

    def __iter__(self):
        bg_timer = Timer(verbose=False, unit=self.unit)
        fg_timer = Timer(verbose=False, unit=self.unit)

        with ToggleGC(self.gc):
            if self.tqdm:
                it = tqdm(range(self.num), desc=self.label, total=self.num)
            else:
                it = range(self.num)

            for i in it:
                bg_timer.start()
                yield fg_timer
                bg_timer.stop()

                if fg_timer.value is not None:
                    time = fg_timer.value
                else:
                    time = bg_timer.value

                self.values.append(time)
                if self.verbose > 2 and not self.tqdm:
                    self.log.profile(f'  Loop {i}: {time:{self.format}} {self.unit}')

        times = np.array(self.values)
        if self.verbose > 1:
            self.log.profile(f'{self.num} loops, best {times.min():{self.format}} {self.unit} [mean {times.mean():{self.format}} ± {times.std():{self.format}} {self.unit}]')
        elif self.verbose > 0:
            self.log.profile(f'{self.num} loops, best {times.min():{self.format}} {self.unit}')
