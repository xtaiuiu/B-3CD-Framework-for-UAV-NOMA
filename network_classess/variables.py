class Variables:
    def __init__(self, x, p, h):
        self.x = x
        self.p = p
        self.h = h

    def variable_dist(self, var_bar):
        #  compute the absolute distance between self and var_bar
        return self.x_dist(var_bar.x) + self.p_dist(var_bar.p) + self.h_dist(var_bar.h)

    def x_dist(self, var_x):
        dist = .0
        for i in range(len(self.x)):
            for k in range(len(self.x[i])):
                dist += abs(self.x[i][k] - var_x[i][k])
        return dist

    def p_dist(self, var_p):
        dist = .0
        for i in range(len(self.p)):
            for k in range(len(self.p[i])):
                dist += abs(self.p[i][k] - var_p[i][k])
        return dist

    def h_dist(self, var_h):
        return abs(self.h - var_h)
