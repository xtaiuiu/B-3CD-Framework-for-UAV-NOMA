# The B<sup>3</sup>CD Framework for NOMA-UAV - Simulation Code

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)

This repository contains the simulation code for the B<sup>3</sup>CD Framework for NOMA-UAV.



## 🚀 Quick Start
### Installation
```bash
pip install -e .
```

### Run Simulation

To run the simulation, just run the following cmd:
```bash
python main_algorithms/run_simulations.py
```
This will perform the following tasks:
#### 1). Generate Terrestrial Users with Different Distributions
First, we plot 4 kinds of user distributions, and mark the near-user centroids and optimal horizontal UAV positions
![User Distributions](simulation_results/distribution.jpg)

```bash
├── simulation_results
├── ├── horizontal_performance
│   ├──└──user_distributions.py         # User Generation
```

#### 2). Generate a UAWN with Road-Based Distributed Users

Second, we use the road-based distributions to generate a NOMA-enabled UAWN with a set of 40 UEs.

```bash
├── scenarios
├── ├── scenario_creators.py            # Create the UAWN
```

The following figures show the user distribution and density.
![User PDE](simulation_results/UE_road_pde.jpg)

Then the near users and far users are paired by the nearest distance principle, as shown below.
![User Pair](simulation_results/UE_road_pairing.jpg)


#### 3). BCD algorithm for Fixed (x, y)
Third, run the BCD algorithm for fixed UAV's horizontal position (x, y) = (0, 0), and plot the convergence curve of
the BCD algorithm



```bash
├── main_algorithms
├── ├── bcd_algorithm.py            # The BCD algorithm
```

![BCD Convergence](simulation_results/BCD_convergence_curve.png)

#### 4). Bayesian optimization (BO) for Horizontal UAV Deployment
Finally, run the Bayesian optimization (BO) to optimize the horizontal UAV deployment, and plot the convergence curve
of BO.

```bash
├── main_algorithms
├── ├── optimize_uav_position_Bayesian.py            # The B^3CD algorithm
```
![B3CD Convergence](simulation_results/B3CD_convergence_curve.png)
## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first.

