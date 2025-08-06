# Calculate the best horizontal deployment for a given user set
# This code generates four deployment points
import numpy as np

from main_algorithms.optimize_uav_position_Bayesian import optimize_horizontal_Bayesian
from scenarios.scenario_creators import load_scenario
from simulation_results.horizontal_performance.user_distributions import create_ue_set_with_distribution, \
    asymmetric_gaussian_distribution, symmetric_gaussian_distribution, road_based_distribution

if __name__ == '__main__':
    num, radius = 300, 100
    UEs = create_ue_set_with_distribution(num, radius, 0.5, 10, asymmetric_gaussian_distribution)
    sc = load_scenario('road_1000_UEs.pickle')
    sc.UEs = UEs
    sc.reset_scenario()
    sc.plot_scenario_kde()
    f_Bayes, x_Bayes = optimize_horizontal_Bayesian(sc)
    print(f"near_center: = {sc.get_near_UEs_center()}")
    print(f"x_Bayes = {x_Bayes}")
