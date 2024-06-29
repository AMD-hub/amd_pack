from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from aleatory.path.path import StochasticProcessState,StochasticProcessPath

# Define the normal distribution function
def normal_pdf(x, mean, std):
    """
    Calculate the probability density function of a normal distribution.

    :param x: The point at which the density is evaluated.
    :param mean: The mean of the normal distribution.
    :param std: The standard deviation of the normal distribution.
    :return: The probability density at point x.
    """
    return norm.pdf(x, mean, std)

class Transition(ABC):
    """
    Abstract base class representing a generic transition mechanism.
    This class defines the interface for evaluating transition densities 
    and simulating the next state in a stochastic process.
    """
    def __init__(self):
        """
        Initialize a Transition object.
        This constructor is called automatically for any subclass of Transition.
        """

        self.minprob = 0.0001

    @abstractmethod
    def density(self, x0: float, xt: Union[float, np.ndarray], t0: float, dt: float) -> Union[float, np.ndarray]:
        """
        Evaluate the transition density at the given arguments.

        :param x0: The current value (initial state).
        :param xt: The target value (final state), can be a float or array.
        :param t0: The initial time.
        :param dt: The time step between x0 and xt.
        :return: The probability density (same dimension as x0 and xt).
        """
        raise NotImplementedError

    @abstractmethod
    def next(self, state : StochasticProcessState, dt: float) -> np.ndarray:
        """
        Simulate the next state based on the current state and time step.

        :param x0: Array of current values (initial states).
        :param t0: The initial time.
        :param dt: The time step between x0 and the next state.
        :return: Array of simulated next states (same dimension as x0).
        """
        raise NotImplementedError

class Transition1D(Transition):
    """
    A one-dimensional implementation of the Transition class.
    This class extends the Transition class for 1D stochastic processes.
    """
    def __init__(self):
        super().__init__()

    def density(self, x0: float, xt: float | np.ndarray, t0: float, dt: float) -> float | np.ndarray:
        """
        Evaluate the transition density for 1D processes.

        :param x0: The current value (initial state).
        :param xt: The target value (final state), can be a float or array.
        :param t0: The initial time.
        :param dt: The time step between x0 and xt.
        :return: The probability density (same dimension as x0 and xt).
        """
        return super().density(x0, xt, t0, dt)
    
    def negLogLikeLihood(self,path : StochasticProcessPath) :
        """
        Evaluate the negative log likelihood for 1D processes.

        :param path: the path of the data. 
        :return: The negative log likelihood.
        """
        x_path  = path.values[:,0,0]
        t_path  = path.time
        dt_path = path.dt
        return -np.sum([
                np.log(max(self.minprob,self.density(x_path[i],x_path[i+1],t_path[i],dt_path[i]))) for i in range(len(x_path)-1)
            ])

    def next(self, state : StochasticProcessState , dt: float, dW: StochasticProcessPath) -> StochasticProcessState:
        """
        Simulate the next state for 1D processes.

        :param x0: Array of current values (initial states).
        :param t0: The initial time.
        :param dt: The time step between x0 and the next state.
        :param dW: The randomness term, can be a float or array.
        :return: Array of simulated next states (same dimension as x0).
        """
        return super().next(state, dt)

class Euler1D(Transition1D):
    """
    An implementation of the Euler-Maruyama method for simulating 1D stochastic processes.
    This class provides methods to calculate transition densities and simulate next states
    using the Euler-Maruyama approximation.
    """
    def __init__(self, drift, diffusion):
        """
        Initialize an Euler1D object with specified drift and diffusion functions.

        :param drift: A callable representing the drift term (function of time and state).
        :param diffusion: A callable representing the diffusion term (function of time and state).
        """
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion
        self.density_iterations = 1
        self.next_iterations = 1
        self.keep_positive = False # should be done

    def density(self, x0: float, xt: float | np.ndarray, t0: float, dt: float) -> float | np.ndarray:
        """
        Calculate the transition density using the Euler-Maruyama approximation.

        :param x0: The current value (initial state).
        :param xt: The target value (final state), can be a float or array.
        :param t0: The initial time.
        :param dt: The time step between x0 and xt.
        :return: The probability density (same dimension as x0 and xt).
        """
        def f(dt, t_0, x_0, n):
            if n == 1:
                # Base case: n=1
                mean = self.drift(t_0, x_0) * dt
                std = self.diffusion(t_0, x_0) * np.sqrt(dt)
                return lambda u: normal_pdf(u, x_0 + mean, std)
            else:
                # Recursive case: n > 1
                previous_distribution = f(dt, t_0, x_0, n-1)
                t_n_minus_1 = t_0 + (n-2) * dt
                
                def integrand(u, x):
                    mean = self.drift(t_n_minus_1, u) * dt
                    std = self.diffusion(t_n_minus_1, u) * np.sqrt(dt)
                    return previous_distribution(u) * normal_pdf(x, u + mean, std)
                
                def distribution(x):
                    mean = x_0 + self.drift(t_0, x_0) * dt
                    std = self.diffusion(t_0, x_0) * np.sqrt(dt)
                    integral, _ = quad(lambda u: integrand(u, x), mean - std * 1.96, mean + std * 1.96)
                    return integral

                return distribution
        final_dist = f(dt / self.density_iterations, t0, x0, self.density_iterations)
        return final_dist(xt)
    
    def next(self, state: StochasticProcessState, dt: float, dW: np.ndarray) -> StochasticProcessState:

        if dW.shape != (self.next_iterations,state.dim,state.ns):    
            raise ValueError(f'You shoud input dW of shape : {self.next_iterations,state.dim,state.ns}')  
        bm_paths = dW.copy()       

        t = state.time
        dt_n = dt/self.next_iterations
        paths = np.zeros((1, state.dim, state.ns))
        paths[0, :, :] = state.values

        for i in range(self.next_iterations):
            drift_term =  np.apply_along_axis(lambda x : self.drift(t,x),0,paths[0,:,:])
            diffusion_term = np.array([np.apply_along_axis(lambda x : self.diffusion(t,x),0,paths[0,:,:])])
            diffusion_term_dWt =  np.einsum('ijk,ljk->ilk', np.expand_dims(bm_paths[i, :, :],axis=0), diffusion_term)
            drift_term_dt      = drift_term*dt
            paths = paths + drift_term_dt + diffusion_term_dWt
            t += dt_n
            if self.keep_positive : paths[paths <= 0] = 0
        return StochasticProcessState(time=state.time+dt,values=paths[0,:,:],column_names=state.column_names)
    
    
    def negLogLikeLihood(self, path: StochasticProcessPath):
        return super().negLogLikeLihood(path)



class Milstein1D(Transition1D):
    """
    An implementation of the Milstein method for simulating 1D stochastic processes.
    This class provides methods to calculate transition densities and simulate next states
    using the Milstein approximation.
    """
    def __init__(self, drift, diffusion):
        """
        Initialize a Milstein1D object with specified drift and diffusion.

        :param drift: A callable representing the drift term (function of time and state).
        :param diffusion: A callable representing the diffusion term (function of time and state).
        """
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion
        self.next_iterations = 1
        self.dx = 10**(-4)
        self.keep_positive = False

    def density(self, x0: float, xt: float | np.ndarray, t0: float, dt: float) -> float | np.ndarray:
        """
        Calculate the transition density using the Milstein approximation.

        :param x0: The current value (initial state).
        :param xt: The target value (final state), can be a float or array.
        :param t0: The initial time.
        :param dt: The time step between x0 and xt.
        :return: The probability density (same dimension as x0 and xt).
        """
        raise NotImplementedError("Milstein transition density is not yet developped")

    def next(self, state: StochasticProcessState, dt: float, dW: np.ndarray) -> StochasticProcessState:
        """
        Simulate the next state using the Milstein method.

        :param x0: Array of current values (initial states).
        :param t0: The initial time.
        :param dt: The time step between x0 and the next state.
        :param dW: The randomness term (increment in Wiener process), can be an array.
        :return: Array of simulated next states (same dimension as x0).
        """
        if dW.shape != (self.next_iterations,state.dim,state.ns):    
            raise ValueError(f'You shoud input dW of shape : {self.next_iterations,state.dim,state.ns}')  
        bm_paths = dW.copy()       

        t = state.time
        dt_n = dt/self.next_iterations
        paths = np.zeros((1, state.dim, state.ns))
        paths[0, :, :] = state.values
        for i in range(self.next_iterations):
            drift_term =  np.apply_along_axis(lambda x : self.drift(t,x),0,paths[0,:,:])
            diffusion_term = np.array([np.apply_along_axis(lambda x : self.diffusion(t, x)*(self.diffusion(t, x+self.dx)-self.diffusion(t, x))/self.dx,0,paths[0,:,:])])
            diffusion_derivative_term = np.array([np.apply_along_axis(lambda x : self.diffusion(t,x),0,paths[0,:,:])])
            diffusion_term_dWt =  np.einsum('ijk,ljk->ilk', np.expand_dims(bm_paths[i, :, :],axis=0), diffusion_term)
            diffusion_derivative_term_dW_sqaure = np.einsum('ijk,ljk->ilk', np.expand_dims(bm_paths[i, :, :]**2-dt, axis=0), diffusion_derivative_term)
            drift_term_dt      = drift_term*dt
            paths =  paths + drift_term_dt + diffusion_term_dWt + 0.5 * diffusion_derivative_term_dW_sqaure
            t += dt_n
            if self.keep_positive : paths[paths <= 0] = 0
        return StochasticProcessState(time=state.time+dt,values=paths[0,:,:],column_names=state.column_names)

class Exact1D(Transition1D):
    """
    A class for exact simulation of 1D stochastic processes, where exact density and next step
    functions are provided.
    """
    def __init__(self, next: callable, density: callable):
        """
        Initialize an Exact1D object with specified exact next step and density functions.

        :param next: A callable that computes the next state given current state and time step.
        :param density: A callable that computes the exact transition density.
        """
        super().__init__()
        self.next_iter = next
        self.density_iter = density

    def density(self, x0: float, xt: float | np.ndarray, t0: float, dt: float) -> float | np.ndarray:
        """
        Evaluate the exact transition density.

        :param x0: The current value (initial state).
        :param xt: The target value (final state), can be a float or array.
        :param t0: The initial time.
        :param dt: The time step between x0 and xt.
        :return: The probability density (same dimension as x0 and xt).
        """
        return self.density_iter(x0, xt, t0, dt)

    def next(self, x0: np.ndarray, t0: float, dt: float, dW: float | np.ndarray) -> np.ndarray:
        """
        Simulate the next state using the exact method provided.

        :param x0: Array of current values (initial states).
        :param t0: The initial time.
        :param dt: The time step between x0 and the next state.
        :param dW: The randomness term, can be a float or array.
        :return: Array of simulated next states (same dimension as x0).
        """
        return self.next_iter(x0, t0, dt, dW)
