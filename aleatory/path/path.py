import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
from aleatory.utils.utils import validate_payoff_simple
from typing import List, Union, Tuple, Optional



class StochasticProcessState:
    """
    Represents a stochastic process path with a single time value (float) and
    provides additional methods for conversion and manipulation.

    Methods:
        convert_to_path(self, state: np.ndarray) -> 'StochasticProcessPath':
            Converts a state to a full path assuming the single time value as the end time.
    """

    def __init__(self,
                time: float,
                values: Optional[np.ndarray] = None,
                column_names: Optional[List[str]] = None):
        """
        Initializes the StochasticProcessState.
        
        Parameters:
            time (float): Single time value.
            values (numpy.ndarray): 3D array of values (time x assets x scenarios).
            column_names (list): Names of the columns in the values array.
        """
        if time is not None and values is not None:
            self.time   = float(time)
            self.values = np.array(values) 
        else:
            raise ValueError("Either provide time and values, or dates and values_dates.")
        if self.values.ndim != 2:
            raise ValueError("Values must be a 2D array with shape (assets, scenarios).")
        
        self.column_names = column_names if column_names is not None else [f'Asset {1+i}' for i in range(self.values.shape[0])]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.values.shape

    @property
    def dim(self) -> int:
        return self.values.shape[0]

    @property
    def ns(self) -> int:
        return self.values.shape[1]

    @property
    def values(self) -> np.ndarray:
        """Values of the path."""
        return self._values

    @values.setter
    def values(self, value: np.ndarray):
        if len(value.shape) == 2:
            self._values = value
        else:
            raise ValueError(
                f"Shape property values is not allowed, either provide shape "
                f"(assets = {self.dim},scenarios = {self.ns})"
            )
    
    @property
    def time(self) -> np.ndarray:
        """
        Returns the time points.

        Returns:
            list: time points.
        """
        return self._time

    @time.setter
    def time(self, value: np.ndarray) -> None:
        """
        Sets new time points.

        Parameters:
            value (list): List of new time points.
        """
        self._time = value 

    @property
    def column_names(self) -> List[str]:
        """
        Returns the column names.

        Returns:
            list: Column names.
        """
        return self._column_names

    @column_names.setter
    def column_names(self, names: List[str]) -> None:
        """
        Sets new column names.

        Parameters:
            names (list): List of new column names.
        """
        if len(names) != self.dim:
            raise ValueError("Length of names must match the number of assets.")
        self._column_names = names


class StochasticProcessPath:
    """
    Represents multiple stochastic process paths for various assets and scenarios.

    Attributes:
        dt (float): The time step used in the process.
        time (numpy.ndarray): An array of time points.
        values (numpy.ndarray): A 3D array of values (time x assets x scenarios).
        column_names (list): Names of the assets in the values array.
        dim (int): The number of assets.
        nt (int): The number of time points.
        ns (int): The number of scenarios.
        shape (int,int,int): (nt,dim,ns).

    Methods:
        get(self, axis: int, index: Union[int, List[int]]) -> np.ndarray
            Retrieves a slice of the `values` array along a specified axis.
        
        extract(self, axis: int, index: Union[int, List[int]]) -> 'StochasticProcessPath'
            Extracts a subset of the path along a specified axis and returns a new `StochasticProcessPath` instance.
        
        add(self, axis: int, new_values: np.ndarray, column_names: Optional[List[str]] = None) -> None
            Adds new time points, assets, or scenarios to the path along a specified axis.

        copy(self) -> 'StochasticProcessPath'
            Creates and returns a copy of the current `StochasticProcessPath` instance.
        
        cumulate(self) -> 'StochasticProcessPath'
            Returns a new path where values are the cumulative sum over time.
        
        differentiate(self) -> 'StochasticProcessPath'
            Returns a new path with values as the differences over time.
        
        accum_maximum(self) -> 'StochasticProcessPath'
            Returns a new path with cumulative maximum values over time.
        
        accum_minimum(self) -> 'StochasticProcessPath'
            Returns a new path with cumulative minimum values over time.
        
        accum_mean(self) -> 'StochasticProcessPath'
            Returns a new path with cumulative mean values over time.
        
        apply_payoff(self, payoff: callable, name: str = 'Payoff') -> 'StochasticProcessPath'
            Applies a payoff function to each scenario and returns the results as a new `StochasticProcessPath`.
        
        mean(self) -> 'StochasticProcessPath'
            Returns a new path representing the mean values over scenarios.
        
        std(self) -> 'StochasticProcessPath'
            Returns a new path representing the standard deviation over scenarios.
        
        print_process(self) -> None
            Prints the process as a DataFrame.
        
        plot_process(self, cmap: str = 'viridis', size_sub: int = 5) -> None
            Plots the process paths.
    """
    def __init__(self,
                time: Optional[np.ndarray] = None,
                values: Optional[np.ndarray] = None,
                column_names: Optional[List[str]] = None):
        """
        Initializes the StochasticProcessPath.
        Parameters:
            dt (numpy.ndarray): Time steps.
            time (numpy.ndarray): Array of time points.
            values (numpy.ndarray): 3D array of values (time x assets x scenarios).
            column_names (list): Names of the columns in the values array.
        """
        if time is not None and values is not None:
            self.time   = np.array(time)
            self.dt     = self.time[1:] - self.time[:-1] 
            self.values = np.array(values) 
        else:
            raise ValueError("Either provide time and values, or dates and values_dates.")
        if self.values.ndim != 3:
            raise ValueError("Values must be a 3D array with shape (time, assets, scenarios).")
        
        self.column_names = column_names if column_names is not None else [f'Asset {1+i}' for i in range(self.values.shape[1])]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.values.shape

    @property
    def dim(self) -> int:
        return self.values.shape[1]

    @property
    def nt(self) -> int:
        return self.values.shape[0]

    @property
    def ns(self) -> int:
        return self.values.shape[2]

    @property
    def values(self) -> np.ndarray:
        """Values of the path."""
        return self._values

    @values.setter
    def values(self, value: np.ndarray):
        if len(value.shape) == 3:
            self._values = value
        else:
            if value.shape == (self.nt, self.dim):
                self._values = np.expand_dims(value, 2)
            elif value.shape == (self.nt, self.ns):
                self._values = np.expand_dims(value, 1)
            else:
                raise ValueError(
                    f"Shape property values is not allowed, either provide shape "
                    f"(time = {self.nt}, assets = {self.dim}) or shape (time = {self.nt}, scenarios = {self.ns})"
                )
    
    @property
    def time(self) -> np.ndarray:
        """
        Returns the time points.

        Returns:
            list: time points.
        """
        return self._time

    @time.setter
    def time(self, value: np.ndarray) -> None:
        """
        Sets new time points.

        Parameters:
            value (list): List of new time points.
        """
        self._time = value 
        self.dt     = value[1:] - value[:-1] 

    @property
    def column_names(self) -> List[str]:
        """
        Returns the column names.

        Returns:
            list: Column names.
        """
        return self._column_names

    @column_names.setter
    def column_names(self, names: List[str]) -> None:
        """
        Sets new column names.

        Parameters:
            names (list): List of new column names.
        """
        if len(names) != self.dim:
            raise ValueError("Length of names must match the number of assets.")
        self._column_names = names

    def get(self, axis: int, index: Union[int, List[int]]) -> np.ndarray:
        """
        Returns a specific time point, asset, or scenario based on the axis.

        Parameters:
            axis (int): Axis to access (0: time, 1: assets, 2: scenarios).
            index (Union[int, List[int]]): Index along the specified axis.

        Returns:
            np.ndarray: The specified slice of the values array.
        """
        if axis not in [0, 1, 2]:
            raise ValueError("Axis must be 0 (time), 1 (assets), or 2 (scenarios).")
        
        if isinstance(index, list):
            if axis == 0:
                if min(index) < 0 or max(index) >= self.nt:
                    raise IndexError(f"Time index out of range, max is {self.nt}.")
            elif axis == 1:
                if min(index) < 0 or max(index) >= self.dim:
                    raise IndexError(f"Asset index out of range, max is {self.dim}.")
            elif axis == 2:
                if min(index) < 0 or max(index) >= self.ns:
                    raise IndexError(f"Scenario index out of range, max is {self.ns}.")
        else:
            if axis == 0 and (index < 0 or index >= self.nt):
                raise IndexError(f"Time index out of range, max is {self.nt}.")
            elif axis == 1 and (index < 0 or index >= self.dim):
                raise IndexError(f"Asset index out of range, max is {self.dim}.")
            elif axis == 2 and (index < 0 or index >= self.ns):
                raise IndexError(f"Scenario index out of range, max is {self.ns}.")

        return np.take(self.values, index, axis=axis)

    def extract(self, axis: int, index: Union[int, List[int]]) -> 'StochasticProcessPath':
        """
        Extracts a specific time point, asset, or scenario based on the axis and returns a new StochasticProcessPath object.
        Parameters:
            axis (int): Axis to access (0: time, 1: assets, 2: scenarios).
            index (int): Index along the specified axis.
        Returns:
            StochasticProcessPath: A new StochasticProcessPath object containing the extracted data.
        """
        if axis not in [0, 1, 2]:
            raise ValueError("Axis must be 0 (time), 1 (assets), or 2 (scenarios).")

        if isinstance(index,int) : index = [index] 
        if axis == 0:
            if min(index) < 0 or max(index) >= self.nt:
                raise IndexError(f"Time index out of range, max is {self.nt}.")
        elif axis == 1:
            if min(index) < 0 or max(index) >= self.dim:
                raise IndexError(f"Asset index out of range, max is {self.dim}.")
        elif axis == 2:
            if min(index) < 0 or max(index) >= self.ns:
                raise IndexError(f"Scenario index out of range, max is {self.ns}.")
        return StochasticProcessPath(
            time= self.time if axis!=0 else [ self.time[i] for i in index ],
            values=np.take(self.values, index, axis=axis),
            column_names= self.column_names if axis!=1 else [ self.column_names[i] for i in index ],
        )

    def add(self, axis: int, new_values: np.ndarray, column_names: Optional[List[str]] = None) -> None:
        """
        Adds new time points, assets, or scenarios based on the axis.
        Parameters:
            axis (int): Axis to add to (0: time, 1: assets, 2: scenarios).
            new_values (numpy.ndarray): Values to add along the specified axis.
        """
        if axis == 0:
            if new_values.shape[1:] != self.values.shape[1:]:
                raise ValueError(f"New values must have shape (time, assets {self.dim}, scenarios {self.ns}).")
            self.values = np.concatenate((self.values, new_values), axis=0)
            x = np.linspace(self.time[-1]+self.dt[-1],self.time[-1]+(new_values.shape[0])*self.dt[-1],new_values.shape[0])
            self.time =  np.concatenate([self.time,x],axis=0)
            
        elif axis == 1:
            if new_values.shape[0] != self.values.shape[0] or new_values.shape[2] != self.values.shape[2]:
                raise ValueError(f"New values must have shape (time {self.nt}, assets, scenarios {self.ns}).")
            self.values = np.concatenate((self.values, new_values), axis=1)
            self.column_names += column_names if column_names is not None else [f'Asset {i+self.dim}' for i in range(new_values.shape[1])]
        elif axis == 2:
            if new_values.shape[:2] != self.values.shape[:2]:
                raise ValueError(f"New values must have shape (time {self.nt}, assets {self.dim}, scenarios).")
            self.values = np.concatenate((self.values, new_values), axis=2)

    def copy(self) -> 'StochasticProcessPath':
        """
        Creates a copy of the process.

        Returns:
            StochasticProcessPath: A copy of the StochasticProcessPath instance.
        """
        return StochasticProcessPath(time=self.time.copy(), values=self.values.copy(), column_names=self.column_names.copy())

    def cumulate(self) -> 'StochasticProcessPath':
        """
        Cumulate the path over the time.
        """
        other = self.copy()
        other.values = np.cumsum(self.values,axis=0)
        return other

    def differentiate(self)  -> 'StochasticProcessPath': 
        """
        differentiate the path over the time.
        """
        other = self.copy()
        other.values = np.concatenate([other.values[:1,:,:],np.diff(other.values,axis=0)],axis=0)
        return other

    def accum_maximum(self) -> 'StochasticProcessPath':
        """
        Calculate the cumulative maximum over the time.
        """
        other = self.copy()
        other.values = np.maximum.accumulate(self.values, axis=0)
        return other

    def accum_minimum(self) -> 'StochasticProcessPath':
        """
        Calculate the cumulative maximum over the time.
        """
        other = self.copy()
        other.values = np.minimum.accumulate(self.values, axis=0)
        return other

    def accum_mean(self) -> 'StochasticProcessPath':
        """
        Calculate the cumulative mean over the time.
        """
        other = self.copy()
        cum_sum = np.cumsum(self.values, axis=0)
        counts = np.arange(1, self.nt + 1)[:, np.newaxis, np.newaxis]
        other.values = cum_sum / counts
        return other

    def apply_payoff(self, payoff: callable, name: str = 'Payoff') -> 'StochasticProcessPath':
        """
        Calculate payoff in the path over the time.
        """
        output = [] 
        arr = self.values 
        time = self.time
        validate_payoff_simple(payoff,self.shape[1])
        for t in range(arr.shape[0]) : 
            scen_t = [] 
            for sc in range(arr.shape[2]) :
                scen_t.append(payoff(time[t],arr[t,:,sc]))
            output.append(scen_t)
        u = np.array(output)
        u = np.expand_dims(u,axis=1)
        other = self.copy()
        other.values = u.copy()
        other.column_names = [f'{name} Calculated']
        return other 

    def mean(self) -> 'StochasticProcessPath':
        """
        Calculate mean of the path over the scenarios.
        """
        other = self.copy()
        other.values = np.mean(self.values,axis=2)
        other.column_names =  [ f'mean {x}' for x in other.column_names]
        return other 

    def std(self) -> 'StochasticProcessPath':
        """
        Calculate std of the path over the scenarios.
        """

        other = self.copy()
        other.values = np.std(self.values,axis=2)
        return other 

    def print_process(self) -> None:
        """
        Prints the process as a DataFrame.
        """
        data = self.values.reshape(-1, self.dim * self.ns)
        cols = pd.MultiIndex.from_tuples([(f'Scenario {scenario+1}',f'{asset}')  for scenario in range(self.ns) for asset in self.column_names])
        df = pd.DataFrame(data, columns=cols)
        print(df)

    def plot_process(self, cmap: str = 'viridis', size_sub: int = 5) -> None:

        num_x, num_subfigs, num_lines = self.shape
        num_cols = 3
        num_rows = (num_subfigs + num_cols - 1) // num_cols
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(size_sub*3, size_sub * num_rows))
        axs = axs.flatten()
        titles = [f'Asset {i+1}' for i in range(num_subfigs)]
        for i in range(num_subfigs):
            for j in range(num_lines):
                axs[i].plot(self.values[:, i, j], color=plt.cm.get_cmap(cmap)(j / num_lines))
            axs[i].set_title(titles[i])
            # axs[i].set_xticklabels([])
            # axs[i].set_yticklabels([])
        for i in range(num_subfigs, len(axs)):
            fig.delaxes(axs[i])
        plt.show()

    def append_state(self, state:StochasticProcessState) :
        new_values = np.zeros((1, self.dim, self.ns))
        new_values[0,:,:] = state.values
        if new_values.shape[1:] != self.values.shape[1:]:
            raise ValueError(f"New state values must have shape (1, assets {self.dim}, scenarios {self.ns}).")
        self.values = np.concatenate((self.values, new_values), axis=0)
        self.time =  np.concatenate([self.time,np.array([state.time])],axis=0)

