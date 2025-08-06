class PhysicalNetwork:
    # the physical network is only a collection of network parameters.
    def __init__(self, p_max, b_tot, radius, h_min, h_max, theta_min, theta_max, g_0, alpha, sigma):
        self.p_max = p_max
        self.b_tot = b_tot
        self.radius = radius
        self.h_min = h_min
        self.h_max = h_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.g_0 = g_0
        self.alpha = alpha
        self.sigma = sigma
        