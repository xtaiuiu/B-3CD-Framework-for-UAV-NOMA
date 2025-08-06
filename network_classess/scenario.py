import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from network_classess.variables import Variables


class Scenario:

    def __init__(self, pn, uav, UEs):
        self.pn = pn
        self.uav = uav
        self.UEs = UEs

    def reset_scenario(self):
        """
        Reset the scenario:
        x: the bandwidth of each UE is set to B_tot/(len(self.UEs)/2)
        p: the power of each UE is set to P_max/len(self.UEs)
        h: the height of each UE is set to h_max^2
        theta: the beam-width of the UAV is set to theta_max^2. So always optimize theta and h first.
        (x_u, y_u): the location of the UAV is set to (0, 0)
        :return: None
        """
        pn = self.pn
        for u in self.UEs:
            u.x = pn.b_tot / (len(self.UEs) / 2)
            u.p = pn.p_max / len(self.UEs)
        self.uav.h = pn.h_max ** 2
        self.uav.theta = pn.theta_max ** 2
        self.uav.u_x, self.uav.u_y = 0, 0
        # check network coverage
        assert math.sqrt(self.uav.h) * math.tan(math.sqrt(self.uav.theta)) >= pn.radius + math.sqrt(
            self.uav.u_x ** 2 + self.uav.u_y ** 2)

    def reset_scenario_OMA(self):
        """
        Reset the scenario in OMA mode:
        x: the bandwidth of each UE is set to B_tot/(len(self.UEs))
        p: the power of each UE is set to P_max/len(self.UEs)
        h: the height of each UE is set to h_max^2
        theta: the beam-width of the UAV is set to theta_max^2. So always optimize theta and h first.
        (x_u, y_u): the location of the UAV is set to (0, 0)
        :return: None
        """
        pn = self.pn
        for u in self.UEs:
            u.x = pn.b_tot / (len(self.UEs))
            u.p = pn.p_max / len(self.UEs)
        self.uav.h = pn.h_max ** 2
        self.uav.theta = pn.theta_max ** 2
        self.uav.u_x, self.uav.u_y = 0, 0
        # check network coverage
        assert math.sqrt(self.uav.h) * math.tan(math.sqrt(self.uav.theta)) >= pn.radius + math.sqrt(
            self.uav.u_x ** 2 + self.uav.u_y ** 2)

    def set_x(self, x):
        """
        Set bandwidth for each UE
        :param x: numpy array of length K, where K is the number of near users.
        :return:
        """
        K = int(len(self.UEs)/2)  # K is the number of near users
        for k in range(K):
            self.UEs[k].x = x[k]
        for k in range(K, 2 * K):
            self.UEs[k].x = x[k - K]

    def set_x_OMA(self, x):
        for k in range(len(self.UEs)):
            self.UEs[k].x = x[k]

    def set_p(self, p):
        """
        Set power for each UE
        :param p: numpy array of length K, in which K is the number of UEs.
        :return:
        """
        for k in range(len(self.UEs)):
            self.UEs[k].p = p[k]

    def set_h_theta(self, h, theta):
        self.uav.h = h
        self.uav.theta = theta

    def get_UE_rates(self):
        """
        Get the rate of each UE under the current system parameters
        :return: a numpy array represents the rate of each UE, and a numpy array represents the extended rate of each UE
        """
        K = int(len(self.UEs)/2)  # K is the number of near users
        rates = np.zeros(3*K)
        pn, uav, U = self.pn, self.uav, self.UEs

        for k in range(K):
            rates[k] = U[k].x * np.log1p(pn.g_0 * U[k].tilde_g * U[k].p / (
                    pn.sigma * U[k].x * uav.theta * ((U[k].loc_x - uav.u_x) ** 2 + (U[k].loc_y - uav.u_y) ** 2 + uav.h) ** (pn.alpha / 2)))
            rates[k+K] = U[k+K].x * np.log1p(pn.g_0 * U[k+K].tilde_g * U[k+K].p /(
                    pn.sigma * U[k+K].x * uav.theta * ((U[k+K].loc_x - uav.u_x) ** 2 + (U[k+K].loc_y - uav.u_y) ** 2 + uav.h) ** (pn.alpha / 2)
                    + pn.g_0 * U[k+K].tilde_g * U[k].p))
            rates[k + 2*K] = U[k + K].x * np.log1p(pn.g_0 * U[k].tilde_g * U[k + K].p / (
                    pn.sigma * U[k + K].x * uav.theta * ((U[k + K].loc_x - uav.u_x) ** 2 + (U[k + K].loc_y - uav.u_y) ** 2 + uav.h) ** (
                        pn.alpha / 2)
                    + pn.g_0 * U[k].tilde_g * U[k].p))
        return rates

    def get_UE_rates_OMA(self):
        """
        Get the rate of each UE under the OMA transmission with current system parameters
        :return: a numpy array represents the rate of each UE
        """
        K = len(self.UEs)
        rates = np.zeros(K)
        pn, uav, U = self.pn, self.uav, self.UEs
        for k in range(K):
            rates[k] = U[k].x * np.log1p(pn.g_0 * U[k].tilde_g * U[k].p / (
                pn.sigma * U[k].x * uav.theta * ((U[k].loc_x - uav.u_x) ** 2 + (U[k].loc_y - uav.u_y) ** 2 + uav.h) ** (pn.alpha/2)
            ))
        return rates

    def get_UEs_center(self):
        # get the geometric center of all UEs
        K = len(self.UEs)
        x = np.array([self.UEs[k].loc_x for k in range(K)])
        y = np.array([self.UEs[k].loc_y for k in range(K)])
        return np.mean(x), np.mean(y)

    def get_near_UEs_center(self):
        # get the geometric center of all near UEs
        K = int(len(self.UEs)/2)
        x = np.array([self.UEs[k].loc_x for k in range(K)])
        y = np.array([self.UEs[k].loc_y for k in range(K)])
        return np.mean(x), np.mean(y)

    def plot_scenario(self):
        # plot the scenario. First, plot the circle of radius pn.radius. Then plot the near users and the far uses by different markers.
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = self.pn.radius * np.cos(theta)
        y_circle = self.pn.radius * np.sin(theta)
        plt.plot(x_circle, y_circle, 'k-')
        for u in self.UEs[:int(len(self.UEs)/2)]:
            plt.plot(u.loc_x, u.loc_y, 'bo')
        for u in self.UEs[int(len(self.UEs)/2):]:
            plt.plot(u.loc_x, u.loc_y, 'r*')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Scenario')
        geo_center_x, geo_center_y = self.get_UEs_center()
        plt.plot(geo_center_x, geo_center_y, 'gs')
        plt.axis('equal')
        plt.show()


    def plot_scenario_kde(self, sigma=1.0):
        """
        Plot the user distribution with Gaussian smoothing.
        
        Parameters:
        -----------
        sigma : float, optional
            Standard deviation for Gaussian kernel. Larger values give smoother results.
            Default is 1.0.
        """
        from scipy.ndimage import gaussian_filter
        
        x = np.array([u.loc_x for u in self.UEs])
        y = np.array([u.loc_y for u in self.UEs])
        nbins = self.pn.radius
        # Create a grid for KDE
        xmin, xmax = -self.pn.radius, self.pn.radius
        ymin, ymax = -self.pn.radius, self.pn.radius
        
        # Calculate 2D histogram of points
        H, xedges, yedges = np.histogram2d(
            x, y,
            bins=nbins,
            range=[[-self.pn.radius, self.pn.radius], 
                  [-self.pn.radius, self.pn.radius]]
        )
        
        # Calculate area of each grid cell in square meters
        cell_area = (2 * self.pn.radius / nbins) ** 2
        
        # Apply Gaussian smoothing to the histogram
        smoothed_H = gaussian_filter(H, sigma=sigma)
        
        # Calculate density (users per square meter)
        density = smoothed_H / cell_area
        
        # Plot the smoothed density with rainbow colormap and gaussian interpolation
        plt.figure(figsize=(10, 8))
        mesh = plt.imshow(density.T, 
                        extent=[xmin, xmax, ymin, ymax],
                        origin='lower',
                        aspect='auto',
                        cmap='rainbow',
                        interpolation='gaussian')
        
        # Add color bar with units
        cbar = plt.colorbar(mesh, fraction=0.046, pad=0.04)
        cbar.set_label('Users per square meter')
        
        # Add circle showing the coverage area
        circle = plt.Circle((0, 0), self.pn.radius, 
                          color='black', fill=False, 
                          linestyle='--', linewidth=1)
        plt.gca().add_patch(circle)
        
        # Set aspect ratio and limits
        plt.gca().set_aspect('equal')
        plt.xlim(-self.pn.radius*1.1, self.pn.radius*1.1)
        plt.ylim(-self.pn.radius*1.1, self.pn.radius*1.1)
        plt.grid(True, alpha=0.3)

        
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(f'Smoothed User Distribution (σ={sigma}, Total: {len(self.UEs)} users)')
        plt.xlim(-self.pn.radius*1.1, self.pn.radius*1.1)
        plt.ylim(-self.pn.radius*1.1, self.pn.radius*1.1)
        plt.axis('equal')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_scenario_kde_compare(self, sigma=1.0):
        """
        Plot the user distribution with different colormap and shading combinations
        for comparison.
        
        Parameters:
        -----------
        sigma : float, optional
            Standard deviation for Gaussian kernel. Larger values give smoother results.
            Default is 1.0.
        """
        from scipy.ndimage import gaussian_filter
        import matplotlib.pyplot as plt
        
        x = np.array([u.loc_x for u in self.UEs])
        y = np.array([u.loc_y for u in self.UEs])
        nbins = 80
        
        # Create a grid for KDE
        xmin, xmax = -self.pn.radius, self.pn.radius
        ymin, ymax = -self.pn.radius, self.pn.radius
        
        # Create a grid of points for KDE evaluation
        xi = np.linspace(xmin, xmax, nbins)
        yi = np.linspace(ymin, ymax, nbins)
        xi, yi = np.meshgrid(xi, yi)
        
        # Calculate 2D histogram of points
        H, xedges, yedges = np.histogram2d(
            x, y,
            bins=nbins,
            range=[[xmin, xmax], [ymin, ymax]]
        )
        
        # Calculate area of each grid cell in square meters
        cell_area = (2 * self.pn.radius / nbins) ** 2
        
        # Apply Gaussian smoothing to the histogram
        smoothed_H = gaussian_filter(H, sigma=sigma)
        
        # Calculate density (users per square meter)
        density = smoothed_H / cell_area
        
        # 40 different colormap and interpolation combinations
        styles = [

            
            # Diverging colormaps with different interpolations
            {'cmap': 'coolwarm', 'interp': 'gaussian', 'title': 'Coolwarm (Gaussian)'},  #
            {'cmap': 'RdYlBu_r', 'interp': 'gaussian', 'title': 'RdYlBu_r (Gaussian)'}, #
            
            # # More sequential colormaps
            # {'cmap': 'cividis', 'interp': 'gaussian', 'title': 'Cividis (Gaussian)'},
            # {'cmap': 'YlOrRd', 'interp': 'gaussian', 'title': 'YlOrRd (Gaussian)'},
            # {'cmap': 'YlOrBr', 'interp': 'gaussian', 'title': 'YlOrBr (Gaussian)'},
            # {'cmap': 'YlGnBu', 'interp': 'gaussian', 'title': 'YlGnBu (Gaussian)'},
            #
            # # More colormaps with spline interpolation
            # {'cmap': 'viridis', 'interp': 'spline16', 'title': 'Viridis (Spline16)'},
            # {'cmap': 'plasma', 'interp': 'spline16', 'title': 'Plasma (Spline16)'},
            # {'cmap': 'inferno', 'interp': 'spline16', 'title': 'Inferno (Spline16)'},
            # {'cmap': 'magma', 'interp': 'spline16', 'title': 'Magma (Spline16)'},
            
            # Rainbow colormaps
            {'cmap': 'rainbow', 'interp': 'gaussian', 'title': 'Rainbow (Gaussian)'},  #
            # {'cmap': 'gist_rainbow', 'interp': 'gaussian', 'title': 'Gist Rainbow (Gaussian)'},
            # {'cmap': 'nipy_spectral', 'interp': 'gaussian', 'title': 'Nipy Spectral (Gaussian)'},
            # {'cmap': 'hsv', 'interp': 'gaussian', 'title': 'HSV (Gaussian)'},
            
            # More colormaps with different interpolations
            # {'cmap': 'turbo', 'interp': 'gaussian', 'title': 'Turbo (Gaussian)'},
            # {'cmap': 'jet', 'interp': 'gaussian', 'title': 'Jet (Gaussian)'},
            # {'cmap': 'viridis', 'interp': 'bicubic', 'title': 'Viridis (Bicubic)'},
            # {'cmap': 'plasma', 'interp': 'bicubic', 'title': 'Plasma (Bicubic)'},
            #
            # # More combinations
            # {'cmap': 'inferno', 'interp': 'bicubic', 'title': 'Inferno (Bicubic)'},
            # {'cmap': 'magma', 'interp': 'bicubic', 'title': 'Magma (Bicubic)'},
            # {'cmap': 'coolwarm', 'interp': 'bicubic', 'title': 'Coolwarm (Bicubic)'},
            # {'cmap': 'bwr', 'interp': 'bicubic', 'title': 'BWR (Bicubic)'},
            
            # Final set of combinations
            # {'cmap': 'seismic', 'interp': 'bicubic', 'title': 'Seismic (Bicubic)'},
            {'cmap': 'RdYlBu_r', 'interp': 'bicubic', 'title': 'RdYlBu_r (Bicubic)'}, #
            # {'cmap': 'viridis', 'interp': 'sinc', 'title': 'Viridis (Sinc)'},
            # {'cmap': 'plasma', 'interp': 'sinc', 'title': 'Plasma (Sinc)'},
            
            # Last few combinations
            # {'cmap': 'inferno', 'interp': 'sinc', 'title': 'Inferno (Sinc)'},
            # {'cmap': 'magma', 'interp': 'sinc', 'title': 'Magma (Sinc)'},
            {'cmap': 'coolwarm', 'interp': 'sinc', 'title': 'Coolwarm (Sinc)'},  #
            # {'cmap': 'bwr', 'interp': 'sinc', 'title': 'BWR (Sinc)'}
        ]
        
        # Create figure with subplots (10 rows, 4 columns)
        fig, axes = plt.subplots(1, 5, figsize=(25, 60))
        axes = axes.ravel()
        
        for idx, style in enumerate(styles):
            ax = axes[idx]
            
            # Use imshow with specified colormap and interpolation
            mesh = ax.imshow(density.T, 
                          extent=[xmin, xmax, ymin, ymax],
                          origin='lower',
                          aspect='auto',
                          cmap=style['cmap'],
                          interpolation=style['interp'])
            
            # Add color bar
            cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Users/m²')
            
            # Add title
            ax.set_title(f"{style['title']}\n(σ={sigma})", fontsize=10)
            
            # Add circle showing the coverage area
            circle = plt.Circle((0, 0), self.pn.radius, 
                              color='black', fill=False, 
                              linestyle='--', linewidth=0.8)
            ax.add_patch(circle)
            
            # Set aspect ratio and limits
            ax.set_aspect('equal')
            ax.set_xlim(-self.pn.radius*1.1, self.pn.radius*1.1)
            ax.set_ylim(-self.pn.radius*1.1, self.pn.radius*1.1)
            ax.grid(True, alpha=0.2)
        
        # Adjust layout with more space for the title
        plt.tight_layout(rect=[0, 0.01, 1, 0.98])
        plt.show()
