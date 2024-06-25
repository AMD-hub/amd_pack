import numpy as np
from aleatory.utils.utils import check_positive_integer, reshaping, validate_dimension, validate_corr_matrix,is_valid_corr_matrix,validate_drift_diffusion,Number
import aleatory.path.path as pt  # Assuming pt is a module containing StochasticProcessPath
from aleatory.processes.base import BaseProcess


class DiffusionProcess(BaseProcess):
    def __init__(self, drift, diffusion, bm, dim=1, time_end=1.0):
        super().__init__(bm.rng, time_end)
        self.drift = drift
        self.diffusion = diffusion
        self.bm = bm
        self.dim = dim
        validate_drift_diffusion(self.drift, self.diffusion, self.dim,self.bm.dim)

    def simulate_path(self,X0,dt=None, time=None, column_names=None, num_scenarios=1) : 
        if isinstance(X0,Number) : X0 = np.array([X0])
        num_steps = len(time)
        dt = dt if dt is not None else (time[1] - time[0])

        
        paths = np.zeros((num_steps, self.dim, num_scenarios))
        paths[0, :, :] = X0[:, None]
        bm_paths = self.bm.simulate_path(dt, time, num_scenarios=num_scenarios,increment=True).values

        for i in range(1, num_steps):
            t = time[i-1]
            X_t = paths[i-1, :, :]
            drift_term =  np.apply_along_axis(lambda x : self.drift(t,x),0,X_t)
            diffusion_term = np.apply_along_axis(lambda x : self.diffusion(t,x),0,X_t)
            diffusion_term_dWt =  np.einsum('ijk,ljk->ilk', np.expand_dims(bm_paths[i, :, :],axis=0), diffusion_term)
            drift_term_dt      = drift_term*dt
            paths[i, :, :] = X_t + drift_term_dt + diffusion_term_dWt

        return pt.StochasticProcessPath(dt=dt, time=time, values=paths, column_names=column_names)
    
    def simulate_at(self, time=None, column_names=None, num_scenarios=1) : 
        pass

    def calibrate(self, path):
        pass 
