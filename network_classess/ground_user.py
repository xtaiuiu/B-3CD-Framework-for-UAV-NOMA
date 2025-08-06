class UE:
    def __init__(self, loc_x, loc_y, tilde_r, tilde_g=1.0, x=1.0, p=1.0):
        self.x = x
        self.p = p
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.tilde_r = tilde_r  # requested transmission rate, \tilde{R}_ij in the paper
        self.tilde_g = tilde_g  # small-scale fading
