import pickle

from network_classess.physical_network import PhysicalNetwork
from network_classess.scenario import Scenario
from network_classess.uav import Uav
import numpy as np

from scenarios.UE_creators import create_UE_set


def create_scenario(n_near, network_radius):
    # a scenario consists of: an UAV, a physical network (mainly physical parameters), a set of slices
    # the parameters are the same as that used in "UAV-Enabled Communication Using NOMA"

    # n_near is the number of near users
    # uav parameters
    uav_height = 10
    uav_theta = np.pi/6
    uav = Uav(0, 0, uav_height, uav_theta)

    # network parameters
    p_max = 100
    b_tot = 100
    h_min = 50
    h_max = 500
    theta_min = np.pi/12
    theta_max = np.pi/2.5
    g_0 = 1.42e-4
    alpha = 4
    sigma = 4e-15
    pn = PhysicalNetwork(p_max, b_tot, network_radius, h_min, h_max, theta_min, theta_max, g_0, alpha, sigma)


    # UE parameters
    tilde_R_low, tilde_R_high = 0.5, 10
    UEs = create_UE_set(n_near, network_radius, tilde_R_low, tilde_R_high, plot=False)

    scenario = Scenario(pn, uav, UEs)
    return scenario

def load_scenario(filename):
    with open(filename, 'rb') as file:
        sc = pickle.load(file)
    return sc


def save_scenario(sc, filename):
    with open(filename, 'wb') as file:
        pickle.dump(sc, file)


if __name__ == '__main__':
    scenario = create_scenario(1, 300)
    save_scenario(scenario, 'scenario_2_UEs.pickle')
    zero_var = scenario.scenario_zero_variables()
    var = scenario.scenario_variables()
    print(var.variable_dist(var))
