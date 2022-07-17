import numpy as np

class Heston():
    def __init__(self, nb_simul: int, nb_steps, maturity: float, strike: float, rate: float, S_0: float, rho: float, vbar:float, kappa: float, gamma: float, v0: float):
        self.nb_simul = nb_simul
        self.nb_steps = nb_steps
        self.maturity = maturity
        self.strike = strike
        self.rate = rate
        self.S_0 = S_0
        self.rho = rho
        self.vbar = vbar
        self.kappa = kappa
        self.gamma = gamma
        self.v0 = v0

    ## Price Put Option
    def GenerateHestonPaths(self, mat, K, rate):
        paths_time = self.get_spots_Heston(mat, K, rate)["S"]
        return np.exp(-self.rate* self.maturity) * np.mean(np.maximum(self.strike - paths_time[:,-1], 0))

    ## Generates Heston simulated prices
    def get_spots_Heston(self, mat = None, K = None, rate = None, kappa = None, rho = None, v0 = None, vbar = None, gamma = None):

        if mat and K and rate:
            self.maturity = mat
            self.strike = K
            self.rate = rate

        dt = self.maturity / float(self.nb_steps)
        # nb_steps = np.round(self.maturity * 252).astype(int)

        Z1 = np.random.normal(0.0, 1.0, [self.nb_simul, self.nb_steps])
        Z2 = np.random.normal(0.0, 1.0, [self.nb_simul, self.nb_steps])
        W1 = np.zeros([self.nb_simul, self.nb_steps + 1])
        W2 = np.zeros([self.nb_simul, self.nb_steps + 1])
        V = np.zeros([self.nb_simul, self.nb_steps + 1])
        X = np.zeros([self.nb_simul, self.nb_steps + 1])
        V[:, 0] = self.v0
        X[:, 0] = np.log(self.S_0)
        time = np.zeros([self.nb_steps + 1])

        for i in range(0, self.nb_steps):
            # making sure that samples from normal have mean 0 and variance 1
            if self.nb_simul > 1:
                Z1[:, i] = (Z1[:, i] - np.mean(Z1[:, i])) / np.std(Z1[:, i])
                Z2[:, i] = (Z2[:, i] - np.mean(Z2[:, i])) / np.std(Z2[:, i])
            Z2[:, i] = self.rho * Z1[:, i] + np.sqrt(1.0 - self.rho ** 2) * Z2[:, i]

            W1[:, i + 1] = W1[:, i] + np.power(dt, 0.5) * Z1[:, i]
            W2[:, i + 1] = W2[:, i] + np.power(dt, 0.5) * Z2[:, i]

            # Truncated boundary condition
            V[:, i + 1] = V[:, i] + self.kappa * (self.vbar - V[:, i]) * dt + self.gamma * np.sqrt(V[:, i]) * (W1[:, i + 1] - W1[:, i])
            V[:, i + 1] = np.maximum(V[:, i + 1], 0.0)

            X[:, i + 1] = X[:, i] + (self.rate - 0.5 * V[:, i]) * dt + np.sqrt(V[:, i]) * (W2[:, i + 1] - W2[:, i])
            time[i + 1] = time[i] + dt

        # Compute exponent
        self.spots_MC = np.exp(X)
        self.paths = {"time": time, "S": self.spots_MC}
        return self.paths