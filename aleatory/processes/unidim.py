import numpy as np
from aleatory.utils.utils import Number, increase_array
import aleatory.path.path as pt  # Assuming pt is a module containing StochasticProcessPath
from aleatory.processes.base import BaseProcess
from aleatory.transition.base import Euler1D
from aleatory.path.path import *

class DiffusionProcess1D(BaseProcess):
    """
    Represents a 1-dimensional diffusion process defined by the SDE:
        dX_t = drift(t, X_t) * dt + diffusion(t, X_t) * dW_t

    Attributes:
        drift: Function defining the drift coefficient.
        diffusion: Function defining the diffusion coefficient.
        bm: Brownian motion object.
        method: Method for numerical simulation (Euler1D).
    """
    def __init__(self, drift, diffusion, bm, time_end=1.0):
        """
        Initializes the DiffusionProcess1D object.

        Args:
            drift: Function defining the drift coefficient.
            diffusion: Function defining the diffusion coefficient.
            bm: Brownian motion object.
            time_end: End time of the simulation.
        """
        super().__init__(bm.rng, time_end)
        self.drift = drift
        self.diffusion = diffusion
        self.bm = bm
        self.method = Euler1D(self.drift,self.diffusion)
        if bm.dim >1 : raise ValueError("In one dimensional process class you can use only one dimensional randomness process")

    def simulate_path(self,X0,time=None, column_names=None, num_scenarios=1) : 
        """
        Simulates paths of the diffusion process.

        Args:
            X0: Initial value or array of initial values.
            time: Time points for simulation.
            column_names: Names for columns in the resulting path.
            num_scenarios: Number of simulation scenarios.

        Returns:
            StochasticProcessPath object containing simulated paths.
        """
        if isinstance(X0,Number) : X0 = np.array([X0])
        num_steps = len(time)       
        paths = np.zeros((num_steps, 1, num_scenarios))
        paths[0, :, :] = X0[:, None]
        sub_iter = self.method.next_iterations if hasattr(self.method, 'next_iterations') else 1
        time_new = increase_array(time,sub_iter)
        # bm_paths = self.bm.simulate_path(time, num_scenarios=num_scenarios,increment=True).values
        bm_paths_new = self.bm.simulate_path(time_new, num_scenarios=num_scenarios,increment=True).values
        dt = time[1:] - time[:-1]
        for i in range(1, num_steps):
            t = time[i-1]
            X_t = paths[i-1, :, :]
            state = StochasticProcessState(t,X_t)
            # paths[i, :, :] = self.method.next(state,dt[i-1],bm_paths[i:i+1,:,:]) .values
            paths[i, :, :] = self.method.next(state,dt[i-1],bm_paths_new[sub_iter*(i-1)+1:sub_iter*i+1,:,:]) .values
        return pt.StochasticProcessPath(time=time, values=paths, column_names=column_names)
    
    def calibrate(self, path,method = 'MLE'):
        """
        Placeholder for calibration method.

        Args:
            path: Path data for calibration.
        """
        pass 

    def density(self, x0: float, xt: float | np.ndarray, t0: float, dt: float) -> float | np.ndarray : 
        """
        Calculates the density function of the diffusion process.

        Args:
            x0: Initial value or array of initial values.
            xt: Target value or array of target values.
            t0: Initial time.
            dt: Time increment.

        Returns:
            Density values corresponding to given inputs.
        """
        return self.method.density(x0,xt,t0,dt)



class CKLSProcess(DiffusionProcess1D) :
    """
    Represents a Continuous-time Chan-Karolyi-Longstaff-Sanders process (CKLS) process derived from DiffusionProcess1D.
        dX_t = -\kappa*(x-b) * dt + \sigma*x**\gamma * dW_t

    Attributes:
        kappa: Parameter related to mean reversion.
        b: Parameter related to mean level.
        gamma: Parameter for the diffusion coefficient.
        sigma: Parameter for the diffusion coefficient.
        lambda_: Parameter for additional drift term.
    """
    def __init__(self, kappa, b, gamma, sigma,bm, time_end=1):
        """
        Initializes th CKLSProcess object.

        Args:
            kappa: Parameter related to mean reversion.
            b: Parameter related to mean level.
            gamma: Parameter for the diffusion coefficient.
            sigma: Parameter for the diffusion coefficient.
            lambda_: Parameter for additional drift term.
            bm: Brownian motion object.
            time_end: End time of the simulation.
        """
        drift = lambda t,x : -kappa*(x-b)
        diffusion = lambda t,x : sigma*(x**gamma)
        super().__init__(drift, diffusion, bm, time_end)
        self.kappa  = kappa
        self.sigma  = sigma
        self.b      = b
        self.gamma  = gamma

    @property
    def kappa(self):
        return self._kappa
    @kappa.setter
    def kappa(self, value):
        self.drift = lambda t,x : -value*(x-self.b)
        self._kappa = value

    @property
    def b(self):
        return self._b
    @b.setter
    def b(self, value):
        self.drift = lambda t,x : -self.kappa*(x-value)
        self._b = value

    @property
    def gamma(self):
        return self._gamma
    @gamma.setter
    def gamma(self, value):
        self.diffusion = lambda t,x : self.sigma*(x**value)
        self._gamma = value

    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, value):
        self.diffusion = lambda t,x : value*(x**self.gamma)
        self._sigma = value

    def simulate_path(self, X0, time=None, column_names=None, num_scenarios=1):
        return super().simulate_path(X0, time, column_names, num_scenarios)
    
    def calibrate(self, path):
        return super().calibrate(path)
    
    def density(self, x0: float, xt: float | np.ndarray, t0: float, dt: float) -> float | np.ndarray:
        return super().density(x0, xt, t0, dt)
