import logging

import numpy as np
import pandas as pd
import time

from matplotlib import pyplot as plt

from benchmark_algorithms.benchmark_meta_heuristic.heuristic import NOMA_heuristic
from scenarios.scenario_creators import create_scenario
from main_algorithms.bcd_algorithm import bcd
from benchmark_algorithms.benchmark_OMA.bcd_OMA import bcd_OMA
from benchmark_algorithms.benchmark_fix_allocation.fix_allocation_benchmarks import fixed_power, fixed_bandwidth
def run(n_repests=1):
    step, end, n_points = 10, 100, 8
    n_ues = 25
    start = end - step * (n_points - 1)
    powers = np.arange(start, end + 1, step)
    obj_avg = pd.DataFrame({'NOMA': np.zeros(len(powers)),
                            'OMA': np.zeros(len(powers)),
                            'Fixed_power': np.zeros(len(powers)),
                            'Fixed_band': np.zeros(len(powers)),
                            'SHIO': np.zeros(len(powers))})
    for repeat in range(n_repests):
        t_r = time.perf_counter()
        sc = create_scenario(n_ues, 100)
        print(f" ####################### repeat = {repeat} ##############################")
        df = pd.DataFrame({'NOMA': np.zeros(len(powers)),
                           'OMA': np.zeros(len(powers)),
                           'Fixed_power': np.zeros(len(powers)),
                           'Fixed_band': np.zeros(len(powers)),
                           'SHIO': np.zeros(len(powers)),})
        for i in range(len(powers)):
            print(f" ####################### repeat = {repeat}, power = {powers[i]} ##############################")
            np.set_printoptions(formatter={'float': '{:0.4f}'.format})
            logging.disable(logging.INFO)
            sc.pn.p_max = powers[i]

            # 1. NOMA
            sc.reset_scenario()
            sc.uav.u_x, sc.uav.u_y = sc.get_near_UEs_center()
            df.iloc[i, 0] = bcd(sc, debug=False, tol=1e-6)

            # 2. OMA
            sc.reset_scenario_OMA()
            sc.uav.u_x, sc.uav.u_y = sc.get_UEs_center()
            df.iloc[i, 1] = bcd_OMA(sc, debug=False, tol=1e-6)

            # 3. Fixed_power
            sc.uav.u_x, sc.uav.u_y = sc.get_near_UEs_center()
            sc.reset_scenario()
            df.iloc[i, 2] = fixed_power(sc, debug=False, tol=1e-6)

            # Fixed_band
            sc.reset_scenario()
            df.iloc[i, 3] = fixed_bandwidth(sc, debug=False, tol=1e-6)

            # 4. SHIO
            sc.reset_scenario()
            f_heu, _ = NOMA_heuristic(sc, method='GBO')
            df.iloc[i, 4] = f_heu
        print(f" ############## repeat = {repeat} in {time.perf_counter() - t_r} seconds ######################")
        obj_avg += df
    obj_avg /= n_repests
    obj_avg.to_excel('power_compare_shio_20times.xlsx')

    # plot
    fontsize = 16
    plt.rcParams.update({'font.size': fontsize})
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    obj_avg.plot(ax=ax1, legend=True, lw=2, xlabel='Maximum transmission power (W)', ylabel='Max-min user rate',
                 fontsize=fontsize, style=["r-s", "m-.d", "c-p", 'k-^', 'g:o'], markersize=10, grid=True,
                 markerfacecolor='none')
    plt.show()

def plot():
    obj_avg = pd.read_excel('power_compare_shio_20times.xlsx', index_col=0)
    step, end, n_points = 20, 200, 3
    n_ues = 25
    start = end - step * (n_points - 1)
    # plot
    fontsize = 16
    plt.rcParams.update({'font.size': fontsize})
    fig1, ax1 = plt.subplots()
    obj_avg.columns = [r'$B^3CD$', 'OMA', 'FPA', 'FBA', 'GBO']
    obj_avg.plot(ax=ax1, legend=False, lw=2, xlabel="Maximum transmission power (W)", ylabel='Max-min user rate (Mbps)',
                 fontsize=fontsize, style=["r-s", "m-.d", "c-p", 'k-^', 'g:o'], markersize=10, grid=True,
                 markerfacecolor='none')
    # ax1.set_xticklabels([str(i) for i in range(start - step, end + 1, step)],
    #                     fontsize=fontsize)  # This is wierd, but correct.
    # ax1.set_yticklabels([f'{x * 100:.0f}%' for x in ax1.get_yticks()])
    x_ticks = [30, 40, 50, 60, 70, 80, 90, 100]
    ax1.set_xticks(range(len(obj_avg)))
    ax1.set_xticklabels(x_ticks)
    plt.show()


if __name__ == '__main__':
    run(n_repests=20)
    plot()
