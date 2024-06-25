# Put functions that you could use
import  numpy as np 
from numbers import Number


def increase_array(x,n) :
    output = np.array([np.linspace(x[i], x[i+1], n+1, endpoint=True)[:-1] for i in range(len(x)-1)]).flatten()
    return np.concatenate([output,[x[-1]]])

def reshaping(x):
    return x.transpose(0, 2, 1)

def repeat(arr,dim,ns) :
    arr_reshaped = arr[:, np.newaxis, np.newaxis]
    arr_reshaped = arr[:, np.newaxis, np.newaxis]
    arr_repeated = np.repeat(arr_reshaped, dim, axis=1)
    arr_final = np.repeat(arr_repeated, ns, axis=2)

    return arr_final


def check_positive_integer(n, name=""):
    """Ensure that the number is a positive integer."""
    if not isinstance(n, int):
        raise TypeError(f"{name} must be an integer.")
    if n <= 0:
        raise ValueError(f"{name} must be positive.")

def check_numeric(value, name=""):
    """Ensure that the value is numeric."""
    if not isinstance(value, Number):
        raise TypeError(f"{name} value must be a number.")

def check_positive_number(value, name=""):
    """Ensure that the value is a positive number."""
    check_numeric(value, name)
    if value <= 0:
        raise ValueError(f"{name} value must be positive.")

def validate_dimension(dim):
    """
    Validates if the dimension is a positive integer.

    Parameters
    ----------
    dim : int
        Dimension to be validated.

    Raises
    ------
    ValueError
        If `dim` is not a positive integer.
    """
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"`dim` must be a positive integer, got {dim}.")

def validate_corr_matrix(corr, dim):
    """
    Validates the correlation matrix.

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix to be validated.
    dim : int
        Dimension of the process.

    Returns
    -------
    np.ndarray
        Validated correlation matrix.

    Raises
    ------
    TypeError
        If `corr` is not of type `np.ndarray`.
    ValueError
        If `corr` is not a valid correlation matrix of the specified dimension.
    """
    if not isinstance(corr, (np.ndarray,float)):
        raise TypeError(f"`corr` must be of type `np.ndarray`, not {type(corr)}")
    elif isinstance(corr,float) : 
        return np.identity(dim)*corr
    elif corr.shape != (dim, dim):
        raise ValueError(f"`corr` must be of dimension ({dim}, {dim}), not {corr.shape}")
    elif not is_valid_corr_matrix(corr):
        raise ValueError("Provided correlation matrix is not valid.")
    return corr

def is_valid_corr_matrix(corr):
    """
    Checks if the provided matrix is a valid correlation matrix.

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix to be validated.

    Returns
    -------
    bool
        True if the matrix is a valid correlation matrix, False otherwise.
    """
    return np.allclose(corr, corr.T) and np.all(np.linalg.eigvals(corr) >= 0) and np.all(np.diag(corr) == 1)


def validate_drift_diffusion(drift, diffusion, dimX,dimW, X0=None):
    """
    Validates that the drift and diffusion functions return arrays of the correct shape.

    Parameters
    ----------
    drift : callable
        Drift function to be validated. Should take arguments (t, X_t) and return an array of shape (dim,).
    diffusion : callable
        Diffusion function to be validated. Should take arguments (t, X_t) and return an array of shape (dim,).
    dimX : int
        Dimension of the process.
    dimW : int
        Dimension of the randomness.
    X0 : np.ndarray, optional
        Initial condition of the process. If None, defaults to an array of zeros of shape (dim,).

    Raises
    ------
    ValueError
        If the drift or diffusion functions do not return arrays of the correct shape.
    """
    if X0 is None:
        X0 = np.zeros(dimX)
    
    t_test = 0.0  # Test with time t=0
    X_test = np.full((dimX,), X0)  # Test with initial condition

    drift_output = drift(t_test, X_test)
    diffusion_output = diffusion(t_test, X_test)

    if not isinstance(drift_output, np.ndarray) or drift_output.shape != (dimX,):
        raise ValueError(f"Drift function must return an array of shape ({dimX},). Got {drift_output.shape} instead.")

    if not isinstance(diffusion_output, np.ndarray) or diffusion_output.shape != (dimX,dimW):
        raise ValueError(f"Diffusion function must return an array of shape ({dimX},{dimW}). Got {diffusion_output.shape} instead.")
    
    print("Drift and diffusion functions are valid.")

def validate_payoff_simple(payoff,dimX, X0=None): 
    """
    Validates that the payoff function return arrays of the correct shape.

    Parameters
    ----------
    Payoff : callable
        Payoff function to be validated. Should take arguments (t, X_t) and return an array of shape (1,).
    dimX : int
        Dimension of the process.
    X0 : np.ndarray, optional
        Initial condition of the process. If None, defaults to an array of zeros of shape (dim,).
    Raises
    ------
    ValueError
        If the Payoff function do not return arrays of the correct shape.
    """
    if X0 is None:
        X0 = np.zeros(dimX)
    
    t_test = 0.0  # Test with time t=0
    X_test = np.full((dimX,), X0)  # Test with initial condition

    output = payoff(t_test, X_test)

    if not isinstance(output,Number) and not isinstance(output, np.ndarray) :
        raise ValueError(f"Payoff function must return a float or an array of shape ({dimX},). Got {type(output)} instead.")
    elif isinstance(output,Number) :
        pass
    elif not isinstance(output, np.ndarray) or output.shape != (1,) :
        raise ValueError(f"Payoff function must return a float or an array of shape ({dimX},). Got {output.shape} instead.")

    print("Payoff function is valid.")


