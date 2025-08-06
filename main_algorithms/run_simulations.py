# entry for the simulation of the INFOCOM paper:
# "Computationally Efficient 3D Deployment for NOMA-UAV Networks with Resource Optimization:
# A Bayesian Approach"
import numpy as np
from matplotlib import pyplot as plt

from main_algorithms.bcd_algorithm import bcd
from main_algorithms.optimize_uav_position_Bayesian import optimize_horizontal_Bayesian
from scenarios.scenario_creators import create_scenario
from simulation_results.horizontal_performance.user_distributions import create_four_cells, plot_UE_set_kde, \
    create_ue_set_with_distribution, road_based_distribution

# 1. First, we plot 4 kinds of user distributions, and mark the near-user centroids and optimal horizontal UAV positions
print(f"1. Generating users...")
UE_set = create_four_cells(100, 3000)
plot_UE_set_kde(UE_set, 100)


# 2. Second, we use the road-based distributions to generate a set of 80 UEs.
print(f"2. Creating UAWN...")
n_ues = 40
sc = create_scenario(n_ues, 100)
UE_set_asymmetric = create_ue_set_with_distribution(round(n_ues / 4) * 4, 100, 0.5, 10, road_based_distribution, plot=True)
sc.UEs = UE_set_asymmetric

# 3. Third, run the BCD algorithm for fixed UAV's horizontal position (x, y) = (0, 0), and plot the convergence curve of
# the BCD algorithm
print(f"3. Run BCD for fixed (x, y) = (0, 0)...")
f_values = bcd(sc, tol=1e-4, debug=True)
extended = np.concatenate([f_values, np.full(46, f_values[3])])
x = np.arange(len(extended))
plt.plot(extended, 'b-o', linewidth=0.5)
plt.title("Convergence curve of BCD", fontsize=14)
plt.xlabel("Iterations", fontsize=12)
plt.ylabel("Objective value", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.ylim(min(extended)*0.95, max(extended)*1.05)

plt.tight_layout()
plt.show()

# 4. Finally, run the Bayesian optimization (BO) to optimize the horizontal UAV deployment, and plot the convergence curve
#  of BO.
print(f"4. Run B^3CD for horizontal UAV deployment...")
f_Bayes, x_Bayes = optimize_horizontal_Bayesian(sc)
