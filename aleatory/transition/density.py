import numpy as np
from typing import Union
from scipy.stats import norm
from scipy.integrate import quad

# Define the normal distribution function
def normal_pdf(x, mean, std):
    return norm.pdf(x, mean, std)

class TransitionDensity:
    def __init__(self, drift,diffusion,iterations = 1):
        """
        Class which represents the transition density for a model, and implements a __call__ method to evalute the
        transition density (bound to the model)

        :param model: the SDE model, referenced during calls to the transition density
        """
        self.drift = drift
        self.diffusion = diffusion
        self.exact_density = None 
        self.iterations = iterations
        self.grid     = [0,100,100]

    def exact(self,
                x0: Union[float, np.ndarray],
                xt: Union[float, np.ndarray],
                t0: Union[float, np.ndarray],
                dt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        The transition density evaluated at these arguments
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        if self.exact_density is None : 
            raise ValueError("The exact density is not assigned you can't call it") 
        else : 
            return self.exact_density(x0,xt,t0,dt) 

    def euler(self,
                x0: Union[float, np.ndarray],
                xt: Union[float, np.ndarray],
                t0: Union[float, np.ndarray],
                dt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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
                    integral, _ = quad(lambda u: integrand(u, x), mean - std*1.96  , mean + std*1.96)
                    return integral

                return distribution
        final_dist = f(dt/self.iterations,t0,x0,self.iterations)
        return final_dist(xt)


    def Lo(self,
                x0: Union[float, np.ndarray],
                xt: Union[float, np.ndarray],
                t0: Union[float, np.ndarray],
                dt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass