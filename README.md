# *aleatory*

here some useful links.... 


## Overview

The **_aleatory_** (/ˈeɪliətəri/) Python library provides functionality for simulating and visualising
stochastic processes. More precisely, it introduces objects representing a number of continuous-time
stochastic processes $X = (X_t : t\geq 0)$ and provides methods to:

- generate realizations/trajectories from each process —over discrete time sets
- create visualisations to illustrate the processes properties and behaviour
- do statistics for these processes like estimations, inference.... 




Currently, `aleatory` supports the following processes:

- Brownian Motion
- Brownian Bridge
- Brownian Excursion
- Brownian Meander
- Geometric Brownian Motion
- Ornstein–Uhlenbeck
- Vasicek
- Cox–Ingersoll–Ross
- Constant Elasticity
- Bessel Process
- Squared Bessel Processs

## Installation

## Dependencies

Aleatory relies heavily on

- ``numpy``  for random number generation
- ``numba``  for heavily calculations
- ``scipy`` and ``statsmodels`` for inference and stats.
- ``matplotlib`` for creating visualisations

## Compatibility

Aleatory is tested on Python versions 3.8, 3.9, 3.10, and 3.11

## Quick-Start

Aleatory allows you to create fancy visualisations from different stochastic processes in an easy and concise way.

For example, the following code

```python
from aleatory.processes import BrownianMotion

brownian = BrownianMotion()
brownian.draw(n=100, N=100, colormap="cool", figsize=(12,9))

```

