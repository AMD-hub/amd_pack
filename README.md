# *amd-pack*

## Overview

The **_amd-pack_** Python library provides functionality for simulating, calibrating and visualising
stochastic processes. More precisely, it introduces objects representing a number of continuous-time
stochastic processes $X = (X_t : t\geq 0)$ and provides methods to:

- generate realizations/trajectories from each process —over discrete time sets
- create visualisations to illustrate the processes properties and behaviour
- do statistics for these processes like estimations, inference.... 




Currently, `aleatory` supports the following processes:

- Brownian Motion
- Diffusion processs (1dim) including :
    - General framework for developpement
    - Chan-Karolyi-Longstaff-Sanders
    - Geometric Brownian Motion
    - Vasicek
    - Cox–Ingersoll–Ross
- Diffusion processs (multivariate) including :
    - General framework for developpement

## Installation

```python
pip install https://github.com/AMD-hub/amd_pack/archive/refs/heads/main.zip
```

## Dependencies

amd-pack relies heavily on

- ``numpy``  for random number generation
- ``scipy`` and ``statsmodels`` for inference and stats.
- ``matplotlib`` for creating visualisations.
- ``pandas`` for data processing.

## Compatibility

amd-pack is tested on Python versions 3.8, 3.9, 3.10, and 3.11


## Quick Start Guide

### Steps to Follow

#### 1. Importing Packages
First, import the necessary modules from the `aleatory` package:
```python
from aleatory.path.path import *
from aleatory.processes.base import *
from aleatory.processes.randomness import *
from aleatory.processes.unidim import *
from aleatory.transition.base import *
```

#### 2. Defining the Time Set
Define the time set over which the model will be simulated:
```python
import numpy as np

years = 5
time = np.linspace(0, years, num=365 * years)
```

#### 3. Creating the Model
Create a simple Brownian Motion and a CKLS process:
```python
bm = BrownianMotion(time_end=years)
# A simple CKLS process
model = CKLSProcess(kappa=0.18, b=0.05, sigma=0.16, gamma=1/2, bm=bm)
```

#### 4. Simulating the Model
Simulate the model using both Euler and Milstein discretization methods:

- **Euler Discretization**:
```python
data = model.simulate_path(X0=0.034, time=time, num_scenarios=30)
data.plot_process()
```

- **Milstein Discretization**:
```python
model.method = Milstein1D(model.drift, model.diffusion)
data = model.simulate_path(X0=0.034, time=time, num_scenarios=30)
data.plot_process()
```

#### 5. Creating a Data Sample for Calibration
Generate a single scenario data sample to calibrate:
```python
data = model.simulate_path(X0=0.034, time=time, num_scenarios=1)
```

#### 6. Calculating Likelihood
Calculate the negative log-likelihood using the Euler method:
```python
model.method = Euler1D(model.drift, model.diffusion)
likelihood = model.negLogLikeLihood(data)
```

#### 7. Calibrating the Model
Calibrate the model using Maximum Likelihood Estimation (MLE):
```python
model.calibrate(data, method='MLE')
```

### Conclusion
By following these steps, you can quickly set up and run a stochastic model, simulate data, and calibrate the model using `aleatory`. This guide provides a foundation for exploring more complex stochastic processes and calibration techniques.

---

You can use this guide as a starting point and modify it as needed to fit your specific requirements or to include more detailed explanations and advanced topics.
