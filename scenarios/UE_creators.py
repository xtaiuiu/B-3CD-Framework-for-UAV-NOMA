import datetime
import math
import random
import matplotlib.pyplot as plt
import pickle

import numpy as np

from network_classess.ground_user import UE


def generate_random_circular_coordinates(R, num_points, plot=False):
    """
    Generate random circular coordinates within a circle of radius R.
    The near users are located in the circle of radius R/2, while the far users are located in the ring with
    radius of R*2/3 and R.

    Then sort the far user, such that the k-th far user is paired with the k-th near user with minimum distance principle.

    :param R: The radius of the circle.
    :param num_points: The number of near and far points to generate.
    :return: a tuple, each of which is a two-dimensional ndarray. They are the coordinates of the near users,
    and the coordinates of the far users.
    """
    # coordinates of the near user
    r_near = R * np.random.rand(num_points)/2
    theta_near = 2 * math.pi * np.random.rand(num_points)
    x_near = r_near * np.cos(theta_near)
    y_near = r_near * np.sin(theta_near)

    # coordinates of the far user
    r_far = R * (2+np.random.rand(num_points))/3
    theta_far = 2 * math.pi * np.random.rand(num_points)
    x_far = r_far * np.cos(theta_far)
    y_far = r_far * np.sin(theta_far)

    # Calculate all possible pairs and their distances
    pairs = []
    for i in range(num_points):  # Near users
        for j in range(num_points):  # Far users
            dx = x_near[i] - x_far[j]
            dy = y_near[i] - y_far[j]
            dist = dx ** 2 + dy ** 2  # Using squared distance for efficiency
            pairs.append((dist, i, j))

    # Sort all possible pairs by distance
    pairs.sort()

    # Greedily select pairs, ensuring each near and far user is only used once
    used_near = set()
    used_far = set()
    pairings = [None] * num_points  # Will store far user index for each near user

    for dist, i, j in pairs:
        if i not in used_near and j not in used_far:
            pairings[i] = j
            used_near.add(i)
            used_far.add(j)
            if len(used_near) == num_points:  # All users paired
                break

    # Reorder far users based on pairings
    x_far = x_far[pairings]
    y_far = y_far[pairings]

    if plot:

        # plot the circular area and the ring
        theta = np.linspace(0, 2*math.pi, 100)
        fig, ax = plt.subplots()
        ax.plot(R/2 * np.cos(theta), R/2 * np.sin(theta), 'k-')
        ax.plot(R * np.cos(theta), R * np.sin(theta), 'k-')
        ax.set_aspect('equal')
        ax.set_title('users distribution')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # plot all users by depicting their locations
        ax.scatter(x_near, y_near, marker='o', c='b', label='near user')
        ax.scatter(x_far, y_far, marker='x', c='r', label='far user')
        for i in range(num_points):
            ax.arrow(x_near[i], y_near[i], x_far[i]-x_near[i], y_far[i]-y_near[i], head_width=0.05, head_length=0.1, fc='k', ec='k', length_includes_head=True)
        ax.legend()
        plt.show()

    return x_near, x_far, y_near, y_far


def create_UE_set(num, network_radius, tilde_R_low, tilde_R_high, plot=False):
    # generate random positions of the UEs
    # tilde_R: the requested rate range of each UE.
    UE_set = []
    x_near, x_far, y_near, y_far = generate_random_circular_coordinates(network_radius, num, plot)
    for i in range(num*2):
        if i < num:
            x = x_near[i]
            y = y_near[i]
        else:
            x = x_far[i-num]
            y = y_far[i-num]
        UE_set.append(UE(x, y, np.random.uniform(tilde_R_low, tilde_R_high)))
    return UE_set


def save_UE(UE_set):
    # 获取当前日期和时间
    now = datetime.datetime.now()

    # 格式化日期和时间，例如：YYYY-MM-DD_HH-MM-SS
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # 定义文件的基本名称和扩展名
    base_filename = "UE_set"
    extension = ".pkl"

    # 组合完整的文件名
    filename = f"{base_filename}_{timestamp}{extension}"
    with open(filename, 'wb') as f:
        pickle.dump(UE_set, f)


def load_UE(filename):
    # 打开包含序列化对象的文件
    with open(filename, 'rb') as f:
        # 使用pickle.load()方法从文件中加载对象
        UE_set = pickle.load(f)
    return UE_set


if __name__ == '__main__':
    network_radius = 3
    create_UE_set(100, network_radius, 0.1, plot=True)