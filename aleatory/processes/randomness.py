import numpy as np
from aleatory.utils.utils import reshaping, validate_dimension, validate_corr_matrix, repeat
import aleatory.path.path as pt  # Assuming pt is a module containing StochasticProcessPath
from aleatory.processes.base import BaseProcess
from typing import List, Union, Tuple, Optional


class WhiteNoise(BaseProcess):
    """
    Class for defining, simulating, and testing a white noise process.
    
    Attributes
    ----------
    cov : float or np.ndarray
        Covariance matrix of the white noise process.
    dim : int
        Dimension of the white noise process.
    """

    def __init__(self, corr: np.ndarray = 1.0, dim: int = 1, rng: Optional[np.random.Generator] = None, time_end: float = 1.0):
        """
        Initializes the WhiteNoise process with specified covariance, dimension, random number generator, and end time.

        Parameters
        ----------
        corr : np.ndarray
            Correlation matrix of the white noise process.
        dim : int, optional
            Dimension of the white noise process. Default is 1.
        rng : np.random.Generator, optional
            Random number generator. Default is None, which uses the default random generator.
        time_end : float, optional
            End time of the process. Default is 1.0.
        """
        super().__init__(rng, time_end)
        
        validate_dimension(dim)
        self.dim = dim 
        self.corr = validate_corr_matrix(corr, dim)

    def simulate_path(self, time: Optional[np.ndarray] = None, column_names: Optional[List[str]] = None, num_scenarios: int = 1) -> pt.StochasticProcessPath:
        """
        Simulates a path of the white noise process.

        Parameters
        ----------
        time : np.ndarray, optional
            Array of time points.
        column_names : list of str, optional
            Names of the columns in the output.
        num_scenarios : int, optional
            Number of scenarios to simulate. Default is 1.

        Returns
        -------
        pt.StochasticProcessPath
            Simulated path of the white noise process.
        """
        aa = reshaping(np.random.multivariate_normal(mean = np.zeros(self.dim) ,cov=self.corr, size=(len(time), num_scenarios)))
        aa[0,:,:] = np.zeros_like(aa[0,:,:])
        return pt.StochasticProcessPath(time=time, values=aa, column_names=column_names)

    def calibrate(self, path: pt.StochasticProcessPath) -> None:
        """
        Calibrates the white noise process to a given path.

        Parameters
        ----------
        path : pt.StochasticProcessPath
            Path of the stochastic process to calibrate to.

        Raises
        ------
        TypeError
            If the path contains more than one scenario.
        """
        if path.ns != 1:
            raise TypeError(f"You are cheating, usually you have a path of only one scenario, not {path.shape}")
        else:
            corr_hat = np.corrcoef(path.values[:,:,0].T)
            self.corr = corr_hat

class BrownianMotion(BaseProcess):
    """
    Class for defining, simulating, and testing a Brownian Motion process.
    
    Attributes
    ----------
    cov : float or np.ndarray
        Covariance matrix of the white brwonian motion.
    dim : int
        Dimension of the white brwonian motion.
    """

    def __init__(self, corr: np.ndarray = None, dim: int = 1, rng: Optional[np.random.Generator] = None, time_end: float = 1.0):
        """
        Initializes the Whitebrwonian motion with specified covariance, dimension, random number generator, and end time.

        Parameters
        ----------
        corr : np.ndarray
            Correlation of the white brwonian motion. If a float is provided, it is multiplied by the identity matrix. Default is 1.0.
        dim : int, optional
            Dimension of the white brwonian motion. Default is 1.
        rng : np.random.Generator, optional
            Random number generator. Default is None, which uses the default random generator.
        time_end : float, optional
            End time of the process. Default is 1.0.
        """
        super().__init__(rng, time_end)
        validate_dimension(dim)
        self.dim = dim 
        if corr is None : corr = np.identity(dim)
        self.corr = validate_corr_matrix(corr, dim)

    def simulate_path(self, time: Optional[np.ndarray] = None, column_names: Optional[List[str]] = None, num_scenarios: int = 1, increment: bool = False) -> pt.StochasticProcessPath:
        """
        Simulates a path of the white brwonian motion.

        Parameters
        ----------
        time : np.ndarray, optional
            Array of time points.
        column_names : list of str, optional
            Names of the columns in the output.
        num_scenarios : int, optional
            Number of scenarios to simulate. Default is 1.

        Returns
        -------
        pt.StochasticProcessPath 
            Simulated path of the white brwonian motion.
        """
        wt = WhiteNoise(corr=self.corr,dim=self.dim,rng=self.rng,time_end=self.t)
        if increment : 
            path        = wt.simulate_path(time,column_names,num_scenarios)
            Delta_t     = np.concatenate([np.array([0]),path.dt])
            path.values = path.values*np.sqrt(repeat(Delta_t,path.dim,path.ns)) 
        else : 
            path    = wt.simulate_path(time,column_names,num_scenarios)
            Delta_t = np.concatenate([np.array([0]),path.dt])
            path.values = path.values*np.sqrt(repeat(Delta_t,path.dim,path.ns)) 
            path    = path.cumulate()
        return path
    
    def calibrate(self, path: pt.StochasticProcessPath) -> None:
        """
        Calibrates the white brwonian motion to a given path.

        Parameters
        ----------
        path : pt.StochasticProcessPath
            Path of the stochastic process to calibrate to.

        Raises
        ------
        TypeError
            If the path contains more than one scenario.
        """

        if path.ns != 1:
            raise TypeError(f"You are cheating, usually you have a path of only one scenario, not {path.ns}")
        else:
            corr_hat = np.corrcoef(path.differentiate().values[:,:,0].T)
            self.corr = corr_hat