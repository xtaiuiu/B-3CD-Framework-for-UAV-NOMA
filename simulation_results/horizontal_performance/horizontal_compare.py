# compare the performance of NOMA and OMA under two different user distributions
# User distribution 1: asymmetric distribution; 2: road_based distribution
# Horizontal UAV deployment 1: near user center; 2: cell center; 3: user center; 4: Bayesian optimization
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from benchmark_algorithms.benchmark_OMA.bcd_OMA import bcd_OMA
from benchmark_algorithms.benchmark_OMA.horizontal_Bayesian_OMA import optimize_horizontal_Bayesian_OMA
from main_algorithms.bcd_algorithm import bcd
from main_algorithms.optimize_uav_position_Bayesian import optimize_horizontal_Bayesian
from scenarios.scenario_creators import load_scenario, create_scenario
from simulation_results.horizontal_performance.user_distributions import create_ue_set_with_distribution, \
    asymmetric_gaussian_distribution, road_based_distribution


def asymmetric_distribution_compare(n_runs=1):
    step, end, n_points = 20, 200, 7
    start = end - step * (n_points - 1)
    n_ues = np.arange(start, end + 1, step)
    obj_avg = pd.DataFrame({'NOMA_cell_center': np.zeros(len(n_ues)),
                            'NOMA_user_center': np.zeros(len(n_ues)),
                            'NOMA_near_center': np.zeros(len(n_ues)),
                            'NOMA_opt_center': np.zeros(len(n_ues)),
                            'OMA_cell_center': np.zeros(len(n_ues)),
                            'OMA_user_center': np.zeros(len(n_ues)),
                            'OMA_near_center': np.zeros(len(n_ues)),
                            'OMA_opt_center': np.zeros(len(n_ues))})
    for repeat in range(n_runs):
        t_r = time.perf_counter()
        print(f" ####################### repeat = {repeat} ##############################")
        df = pd.DataFrame({'NOMA_cell_center': np.zeros(len(n_ues)),
                           'NOMA_user_center': np.zeros(len(n_ues)),
                           'NOMA_near_center': np.zeros(len(n_ues)),
                           'NOMA_opt_center': np.zeros(len(n_ues)),
                           'OMA_cell_center': np.zeros(len(n_ues)),
                           'OMA_user_center': np.zeros(len(n_ues)),
                           'OMA_near_center': np.zeros(len(n_ues)),
                           'OMA_opt_center': np.zeros(len(n_ues))})
        for i in range(len(n_ues)):
            print(f" ####################### repeat = {repeat}, N = {n_ues[i]} ##############################")
            np.set_printoptions(formatter={'float': '{:0.4f}'.format})
            sc = create_scenario(n_ues[i], 100)
            UE_set_asymmetric = create_ue_set_with_distribution(round(n_ues[i] / 4) * 4, 100, 0.5, 10, road_based_distribution, plot=False)
            sc.UEs = UE_set_asymmetric

            sc.reset_scenario()
            sc.uav.u_x, sc.uav.u_y = 0, 0  # cell center
            df.iloc[i, 0] = bcd(sc)

            sc.reset_scenario()
            sc.uav.u_x, sc.uav.u_y = sc.get_UEs_center()  # user center
            df.iloc[i, 1] = bcd(sc)

            sc.reset_scenario()
            sc.uav.u_x, sc.uav.u_y = sc.get_near_UEs_center()  # near center
            df.iloc[i, 2] = bcd(sc)

            sc.reset_scenario()
            df.iloc[i, 3], _ = optimize_horizontal_Bayesian(sc)  # Bayesian optimization

            sc.reset_scenario_OMA()
            sc.uav.u_x, sc.uav.u_y = 0, 0  # cell center
            df.iloc[i, 4] = bcd_OMA(sc)

            sc.reset_scenario_OMA()
            sc.uav.u_x, sc.uav.u_y = sc.get_UEs_center()  # user center
            df.iloc[i, 5] = bcd_OMA(sc)

            sc.reset_scenario_OMA()
            sc.uav.u_x, sc.uav.u_y = sc.get_near_UEs_center()  # near center
            df.iloc[i, 6] = bcd_OMA(sc)

            sc.reset_scenario_OMA()
            df.iloc[i, 7], _ = optimize_horizontal_Bayesian_OMA(sc)  # Bayesian optimization of OMA
            print(f"df = {df.iloc[i]}")

        print(f" ############## repeat = {repeat} in {time.perf_counter() - t_r} seconds ######################")

        obj_avg += df
    obj_avg /= n_runs
    obj_avg.to_excel('horizontal_compare_road_total_rates.xlsx')


def plot_asymmetric_distribution_compare():
    fontsize = 16
    step, end, n_points = 20, 200, 7
    start = end - step * (n_points - 1)


    plt.rcParams.update({'font.size': fontsize})
    df = pd.read_excel('horizontal_compare_road_total_rates.xlsx', index_col=0)
    df_NOMA = df[['NOMA_cell_center', 'NOMA_user_center', 'NOMA_near_center', 'NOMA_opt_center']].copy()
    df_OMA = df[['OMA_cell_center', 'OMA_user_center', 'OMA_near_center', 'OMA_opt_center']].copy()

    df_NOMA = df_NOMA.apply(lambda x: (df_NOMA['NOMA_opt_center'] - x) / df_NOMA['NOMA_opt_center'])
    df_OMA = df_OMA.apply(lambda x: (df_OMA['OMA_opt_center'] - x) / df_OMA['OMA_opt_center'])

    df_NOMA.columns = ['cell_center', 'user_centroid', 'near_user_centroid', 'Bayesian_opt']
    df_OMA.columns = ['cell_center', 'user_centroid', 'near_user_centroid', 'Bayesian_opt']

    fig1, ax1 = plt.subplots()
    df_NOMA.plot(ax=ax1, legend=True, lw=2, xlabel='Number of users', ylabel='Relative objective gap', fontsize=fontsize, style=["r-s", "m-.d", "c-p", 'k-^'], markersize=10, grid=True, markerfacecolor='none')
    ax1.set_xticklabels([str(i) for i in range(start-step, end + 1, step)], fontsize=fontsize)  # This is wierd, but correct.
    ax1.set_yticklabels([f'{x*100:.0f}%' for x in ax1.get_yticks()])

    fig2, ax2 = plt.subplots()
    df_OMA.plot(ax=ax2, legend=True, lw=2, xlabel='Number of users', ylabel='Relative objective gap', fontsize=fontsize, style=["r-s", "m-.d", "c-p", 'k-^'], markersize=10, grid=True, markerfacecolor='none')
    ax2.set_xticklabels([str(i) for i in range(start-step, end + 1, step)], fontsize=fontsize)  # This is wierd, but correct.
    ax2.set_yticklabels([f'{x*100:.0f}%' for x in ax2.get_yticks()])
    ax2.legend(bbox_to_anchor=(0.5, 0.7), loc='center', borderaxespad=0.)
    plt.show()



if __name__ == '__main__':
    # asymmetric_distribution_compare(n_runs=20)
    plot_asymmetric_distribution_compare()
