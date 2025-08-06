# create scenarios with different user distributions
# plot the user distribution by gaussian kde
import math
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal

from main_algorithms.bcd_algorithm import bcd
from main_algorithms.optimize_uav_position_Bayesian import optimize_horizontal_Bayesian
from network_classess.ground_user import UE
from scenarios.scenario_creators import create_scenario, save_scenario, load_scenario


def uniform_distribution(R, num_points, plot=False):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    r = np.sqrt(np.random.uniform(0, R ** 2, num_points))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].scatter(x, y)
        ax[0].set_aspect('equal')
        ax[0].set_title('Uniform distribution')
        xy = np.vstack([x, y])
        nbins = 40
        xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
        zi = gaussian_kde(xy)(np.vstack([xi.flatten(), yi.flatten()]))
        ax[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.BuGn_r)
        ax[1].set_aspect('equal')
        ax[1].set_title('Uniform distribution with density')
    return np.column_stack((x, y))


def symmetric_gaussian_distribution(R, num_points, plot=False):
    n_grid = 1  # number of grids, which controls the variance of Gaussian
    G1 = multivariate_normal(mean=[R/3, R/3], cov=[[R/n_grid/0.5, 0], [0, R/n_grid/0.5]])
    G2 = multivariate_normal(mean=[-R/3, -R/3], cov=[[R/n_grid/0.5, 0], [0, R/n_grid/0.5]])
    K = int(num_points / 2)
    samples_G1 = []
    for k in range(K):
        sample_G1 = G1.rvs()
        while np.sqrt(sample_G1[0] ** 2 + sample_G1[1] ** 2 ) > R:
            sample_G1 = G1.rvs()
        samples_G1.append(G1.rvs())

    samples_G2 = []
    for k in range(K):
        sample_G2 = G2.rvs()
        while np.sqrt(sample_G2[0] ** 2 + sample_G2[1] ** 2 ) > R:
            sample_G2 = G2.rvs()
        samples_G2.append(G2.rvs())

    samples_G1 = np.array(samples_G1)
    samples_G2 = np.array(samples_G2)
    samples = np.vstack([samples_G1, samples_G2])
    assert samples.shape[0] == num_points
    for i in range(samples.shape[0]):
        assert np.sqrt(samples[i, 0] ** 2 + samples[i, 1] ** 2) <= R + 0.1*R, f'sample out of range, dist = {np.sqrt(samples[i, 0] ** 2 + samples[i, 1] ** 2)}, R = {R}'
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].scatter(samples[:, 0], samples[:, 1])
        ax[0].set_aspect('equal')
        ax[0].set_title('Symmetric Gaussian distribution')
        xy = np.vstack([samples[:, 0], samples[:, 1]])
        nbins = 80
        xi, yi = np.mgrid[samples[:, 0].min():samples[:, 0].max():nbins * 1j, samples[:, 1].min():samples[:, 1].max():nbins * 1j]
        zi = gaussian_kde(xy)(np.vstack([xi.flatten(), yi.flatten()]))
        ax[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='nearest', cmap='Reds')
        ax[1].set_aspect('equal')
        ax[1].set_title('Symmetric Gaussian distribution with density')
    return samples


def asymmetric_gaussian_distribution(R, num_points, plot=False):
    # num_points should be divisible by 3
    n_grid = 1  # number of grids, which controls the variance of Gaussian
    G1 = multivariate_normal(mean=[R / 3, R / 3], cov=[[R / n_grid/0.5, 0], [0, R / n_grid/0.5]])
    G2 = multivariate_normal(mean=[-R / 3, -R / 3], cov=[[R / n_grid/0.5, 0], [0, R / n_grid/0.5]])
    G3 = multivariate_normal(mean=[-R / 3, R / 3], cov=[[R / n_grid/0.5, 0], [0, R / n_grid/0.5]])
    K = int(num_points / 3)
    samples_G1 = []
    for k in range(K):
        sample_G1 = G1.rvs()
        while np.sqrt(sample_G1[0] ** 2 + sample_G1[1] ** 2) > R:
            sample_G1 = G1.rvs()
        samples_G1.append(G1.rvs())

    samples_G2 = []
    for k in range(K):
        sample_G2 = G2.rvs()
        while np.sqrt(sample_G2[0] ** 2 + sample_G2[1] ** 2) > R:
            sample_G2 = G2.rvs()
        samples_G2.append(G2.rvs())

    samples_G3 = []
    for k in range(K):
        sample_G3 = G3.rvs()
        while np.sqrt(sample_G3[0] ** 2 + sample_G3[1] ** 2) > R:
            sample_G3 = G3.rvs()
        samples_G3.append(G3.rvs())

    samples_G1 = np.array(samples_G1)
    samples_G2 = np.array(samples_G2)
    samples_G3 = np.array(samples_G3)
    samples = np.vstack([samples_G1, samples_G2, samples_G3])
    assert samples.shape[0] == num_points
    # for i in range(samples.shape[0]):
    #     assert np.sqrt(samples[i, 0] ** 2 + samples[
    #         i, 1] ** 2) <= R+0.2*R, f'sample out of range, dist = {np.sqrt(samples[i, 0] ** 2 + samples[i, 1] ** 2)}, R = {R}'
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].scatter(samples[:, 0], samples[:, 1])
        ax[0].set_aspect('equal')
        ax[0].set_title('Asymmetric Gaussian distribution')
        xy = np.vstack([samples[:, 0], samples[:, 1]])
        nbins = 80
        xi, yi = np.mgrid[samples[:, 0].min():samples[:, 0].max():nbins * 1j, samples[:, 1].min():samples[:, 1].max():nbins * 1j]
        zi = gaussian_kde(xy)(np.vstack([xi.flatten(), yi.flatten()]))
        ax[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.BuGn_r)
        ax[1].set_aspect('equal')
        ax[1].set_title('Asymmetric Gaussian distribution with density')
    return samples


def road_based_distribution(R, num_points, plot=False):
    # num_points should be divisible by 4
    # Generate two clusters of points.
    # num_points * (2/3) points are in the first cluster, and num_points * (1/3) points are in the second cluster.
    # The first cluster is located within the belt of width R/4. The belt if regulated by two horizontal lines, one is the line y = R/2, and the other is y = 3*R/4.
    # The second cluster is generated by a two dimensional Gaussian distribution, whose mean is (0, -R/2), and the variance is R^2/4.
    K = int(num_points / 4)
    # First cluster
    samples_G1 = []
    for k in range(3*K):
        sample_G1 = [np.random.uniform(-math.sqrt(15)*R/4, math.sqrt(15)*R/4), np.random.uniform(R/2, 3*R/4)]
        while np.sqrt(sample_G1[0] ** 2 + sample_G1[1] ** 2) > R:
            sample_G1 = [np.random.uniform(-math.sqrt(15)*R/4, math.sqrt(15)*R/4), np.random.uniform(R/2, 3*R/4)]
        samples_G1.append(sample_G1)
    samples_G1 = np.array(samples_G1)
    # Second cluster
    G2 = multivariate_normal(mean=[0, -R/4], cov=[[R*2, 0], [0, R*2]])
    samples_G2 = []
    for k in range(K):
        sample_G2 = G2.rvs()
        while np.sqrt(sample_G2[0] ** 2 + sample_G2[1] ** 2) > R:
            sample_G2 = G2.rvs()
        samples_G2.append(G2.rvs())
    samples_G2 = np.array(samples_G2)
    samples = np.vstack([samples_G1, samples_G2])
    assert samples.shape[0] == num_points
    # for i in range(samples.shape[0]):
    #     assert np.sqrt(samples[i, 0] ** 2 + samples[
    #         i, 1] ** 2) <= R, f'sample out of range, dist = {np.sqrt(samples[i, 0] ** 2 + samples[i, 1] ** 2)}, R = {R}'

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].scatter(samples[:, 0], samples[:, 1])
        ax[0].set_xlim(-R, R)
        ax[0].set_ylim(-R, R)
        ax[0].set_aspect('equal')
        ax[0].add_patch(plt.Circle((0, 0), R, color='black', fill=False, linestyle='--'))
        ax[0].set_title('Road-based distribution')
        xy = np.vstack([samples[:, 0], samples[:, 1]])
        nbins = 80
        xi, yi = np.mgrid[-R:R:nbins * 1j, -R:R:nbins * 1j]
        zi = gaussian_kde(xy)(np.vstack([xi.flatten(), yi.flatten()]))
        ax[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='Reds')
        ax[1].set_xlim(-R, R)
        ax[1].set_ylim(-R, R)
        ax[1].set_aspect('equal')
        ax[1].add_patch(plt.Circle((0, 0), R, color='black', fill=False, linestyle='--'))
        ax[1].set_title('Road-based distribution with density')
    return samples


import numpy as np
from scipy.spatial import distance


def pair_near_far_points(network_radius, samples, plot=False):
    """
    将样本点分为近点和远点，并为每个近点配对最近的远点

    参数:
        samples: (N, 2)的ndarray，N为偶数

    返回:
        samples_near: (N/2, 2) 离原点最近的N/2个点
        samples_far: (N/2, 2) 与samples_near对应的最近远点
    """
    N = len(samples)

    # 1. 计算每个点到原点的距离并排序
    dist_from_origin = np.linalg.norm(samples, axis=1)
    sorted_indices = np.argsort(dist_from_origin)

    # 2. 分为近点和远点
    near_indices = sorted_indices[:N // 2]
    far_indices = sorted_indices[N // 2:]

    samples_near = samples[near_indices]
    samples_far_all = samples[far_indices]

    # 3. 为每个近点找到最近的远点
    # 计算所有近点和远点之间的距离矩阵
    dist_matrix = distance.cdist(samples_near, samples_far_all)

    # 初始化
    samples_far = np.zeros_like(samples_near)
    used_far_indices = set()  # 记录已使用的远点索引

    for k in range(len(samples_near)):
        # 找到未使用的最接近的远点
        min_dist = np.inf
        best_far_idx = -1

        for j in range(len(samples_far_all)):
            if j not in used_far_indices and dist_matrix[k, j] < min_dist:
                min_dist = dist_matrix[k, j]
                best_far_idx = j

        # 记录配对结果
        samples_far[k] = samples_far_all[best_far_idx]
        used_far_indices.add(best_far_idx)

    if plot:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

        # 绘制原点
        plt.scatter([0], [0], c='black', s=100, marker='+', label='Origin (0,0)')

        # 绘制近点（红色星号）
        plt.scatter(samples_near[:, 0], samples_near[:, 1],
                    c='red', s=100, marker='*', label='Near Users')

        # 绘制远点（蓝色圆圈）
        plt.scatter(samples_far[:, 0], samples_far[:, 1],
                    c='blue', s=100, marker='o', label='Far Users')

        # plot the circle
        circle = plt.Circle((0, 0), network_radius, color='black', fill=False, linestyle='--')
        ax.add_patch(circle)

        # 添加箭头连线
        for near_pt, far_pt in zip(samples_near, samples_far):
            ax.annotate("", xy=far_pt, xytext=near_pt,
                        arrowprops=dict(arrowstyle="->",
                                        color="green",
                                        linewidth=1,
                                        linestyle="--",
                                        alpha=0.7))

        # 添加标签和标题
        plt.title("Near-Far Point Pairing with Arrows", fontsize=14)
        plt.xlabel("X coordinate", fontsize=12)
        plt.ylabel("Y coordinate", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=12)

        # 等比例坐标轴
        plt.axis('equal')

        # 显示图形
        plt.show()

    return np.vstack([samples_near, samples_far])


def create_ue_set_with_distribution(num, network_radius, tilde_R_low, tilde_R_high, distribution=uniform_distribution, plot=False):
    # num is the number of all users, not the number of near users
    UE_set = []
    samples = distribution(network_radius, num, plot)
    paired_samples = pair_near_far_points(network_radius, samples, plot)
    for j in range(num):
        x = paired_samples[j][0]
        y = paired_samples[j][1]
        UE_set.append(UE(x, y, np.random.uniform(tilde_R_low, tilde_R_high)))
    return UE_set


def create_four_cells(R, num, plot=False):
    # generate four cells
    # R: the radius of the network
    # the center of each cell is (R, R), (R, -R), (-R, -R), (-R, R)
    # the radius of each cell is R
    # the number of UEs in each cell is num
    UE_set_uniform = create_ue_set_with_distribution(num*5, R, 0.5, 10, uniform_distribution)
    UE_set_symmetric = create_ue_set_with_distribution(int(num/1.5), R, 0.5, 10, symmetric_gaussian_distribution)
    UE_set_asymmetric = create_ue_set_with_distribution(num, R, 0.5, 10, asymmetric_gaussian_distribution)
    UE_set_road = create_ue_set_with_distribution(num, R, 0.5, 10, road_based_distribution)


    # move the UEs to the center of each cell
    for j in range(len(UE_set_uniform)):
        UE_set_uniform[j].loc_x += -R
        UE_set_uniform[j].loc_y += R

    for j in range(len(UE_set_road)):
        UE_set_road[j].loc_x += R
        UE_set_road[j].loc_y += -R

    for j in range(len(UE_set_symmetric)):
        UE_set_symmetric[j].loc_x += R
        UE_set_symmetric[j].loc_y += R
    for j in range(num):
        UE_set_asymmetric[j].loc_x += -R
        UE_set_asymmetric[j].loc_y += -R

    UE_set = UE_set_uniform + UE_set_symmetric + UE_set_asymmetric + UE_set_road

    return UE_set


def plot_UE_set_kde(UE_set, radius, sigma=1.5):
    """
    Plot the user distribution with Gaussian smoothing for a given UE set.
    
    Parameters:
    -----------
    UE_set : list
        List of UE objects with x and y attributes
    radius : float
        Radius of the coverage area in meters
    sigma : float, optional
        Standard deviation for Gaussian kernel. Larger values give smoother results.
        Default is 1.5.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    
    # Extract x and y coordinates from UE set
    x = np.array([ue.loc_x for ue in UE_set])
    y = np.array([ue.loc_y for ue in UE_set])
    
    # Set up grid parameters
    nbins = int(radius)  # Number of bins based on radius
    xmin, xmax = -2*radius, 2*radius
    ymin, ymax = -2*radius, 2*radius
    
    # Calculate 2D histogram of points
    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=nbins,
        range=[[-2*radius, 2*radius], [-2*radius, 2*radius]]
    )
    
    # Calculate area of each grid cell in square meters
    cell_area = (2 * radius / nbins) ** 2
    
    # Apply Gaussian smoothing to the histogram
    smoothed_H = gaussian_filter(H, sigma=sigma)
    
    # Calculate density (users per square meter)
    density = smoothed_H / cell_area
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot the smoothed density with rainbow colormap and gaussian interpolation
    mesh = plt.imshow(density.T, 
                    extent=[xmin, xmax, ymin, ymax],
                    origin='lower',
                    aspect='auto',
                    cmap='rainbow',
                    interpolation='gaussian')
    
    # Add color bar with units
    cbar = plt.colorbar(mesh, fraction=0.046, pad=0.04)
    cbar.set_label(r'User density (users/$m^2$)', fontsize=18)

    # Add labels and title
    
    # Add circle showing the coverage area
    circle_uniform = plt.Circle((-radius, radius), radius,
                      color='black', fill=False,
                      linestyle='--', linewidth=0.5)
    circle_symmetric = plt.Circle((radius, radius), radius,
                                color='black', fill=False,
                                linestyle='--', linewidth=0.5)
    circle_asymmetric = plt.Circle((-radius, -radius), radius,
                                color='black', fill=False,
                                linestyle='--', linewidth=0.5)
    circle_road = plt.Circle((radius, -radius), radius,
                                color='black', fill=False,
                                linestyle='--', linewidth=0.5)

    plt.gca().add_patch(circle_uniform)
    plt.gca().add_patch(circle_symmetric)
    plt.gca().add_patch(circle_asymmetric)
    plt.gca().add_patch(circle_road)

    # Plot the near user center and the Bayesian optimized UAV position
    # Uniform distribution
    plt.rcParams.update({'font.size': 18})

    plt.scatter(-radius, radius, marker='o', color='red', s=80, label='near user centroid')
    plt.scatter(-radius, radius, marker='^', color='black', s=100, label='UAV\'s best position')

    # Symmetric distribution
    plt.scatter(radius, radius, marker='o', color='red', s=80)

    plt.scatter(radius, radius, marker='^', color='black', s=100)

    # Asymmetric distribution
    plt.scatter(-radius - 8.3, -radius+8.11, marker='o', color='red', s=80)

    plt.scatter(-radius - 3.76, -radius+2.733, marker='^', color='black', s=100)

    # Road distribution
    plt.scatter(radius+1.447, -radius+17.028, marker='o', color='red', s=80)

    plt.scatter(radius+2.185, -radius+13.3767, marker='^', color='black', s=100)

    plt.legend(loc='upper center')
    
    # Set aspect ratio and limits
    plt.gca().set_aspect('equal')
    plt.xlim(-radius*2.1, radius*2.1)
    plt.ylim(-radius*2.1, radius*2.4)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    UE_set = create_four_cells(100, 3000)
    plot_UE_set_kde(UE_set, 100)
