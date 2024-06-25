from abc import ABC
import numpy as np

class BaseProcess(ABC):
    """
    Base class for stochastic processes.
    
    Properties
    ----------
    - rng : np.random.RandomState
        Randomness generator
    - t : float 
        Domain definition is `[0,t]`
    """

    def __init__(self, rng=None, time_end: float = 1.0):
        self.rng = rng
        self.t = time_end


    @property
    def rng(self):
        if self._rng is None:
            return np.random.default_rng()
        return self._rng

    @rng.setter
    def rng(self, value):
        if value is None:
            self._rng = None
        elif isinstance(value, (np.random.RandomState, np.random.Generator)):
            self._rng = value
        else:
            raise TypeError("rng must be of type `np.random.RandomState`")

    @property
    def t(self):
        """End time of the process."""
        return self._t

    @t.setter
    def t(self, value):
        self._t = float(value)
