import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
import plotly.graph_objects as go
from plotly.graph_objs import Surface
from plotly.offline import iplot, init_notebook_mode


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

    def GenerateHestonPaths(self, mat, K, rate):
        self.maturity = mat
        self.strike = K
        self.rate = rate

        dt = self.maturity / float(self.nb_steps)
        nb_steps = np.round(self.maturity * 252).astype(int)

        Z1 = np.random.normal(0.0, 1.0, [self.nb_simul, nb_steps])
        Z2 = np.random.normal(0.0, 1.0, [self.nb_simul, nb_steps])
        W1 = np.zeros([self.nb_simul, nb_steps + 1])
        W2 = np.zeros([self.nb_simul, nb_steps + 1])
        V = np.zeros([self.nb_simul, nb_steps + 1])
        X = np.zeros([self.nb_simul, nb_steps + 1])
        V[:, 0] = self.v0
        X[:, 0] = np.log(self.S_0)

        time = np.zeros([nb_steps + 1])

        for i in range(0, nb_steps):
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
        S = np.exp(X)
        paths = {"time": time, "S": S[:-1]}
        return np.exp(-self.rate*self.maturity)*np.mean(np.max(S[:,-1] - self.strike, 0))





class Numerics(Heston):
    def __init__(self, nb_simul, nb_steps, maturity, strike, rate, S_0, rho, vbar, kappa, gamma, v0):
        super().__init__(nb_simul, nb_steps, maturity, strike, rate, S_0, rho, vbar, kappa, gamma, v0)

    def market_data(self):
        df = pd.read_csv('/Users/pierreranchet/Documents/Sauvegarde_PC/Dauphine/Cours/S2/Produits_Structures/Projet_Autocall/FTSE_Prices.csv', sep=";")
        self.S_0 = 7208.81
        self.maturity_tot = np.array([(datetime.strptime(i, '%m/%d/%Y') - datetime.today()).days for i in df.Maturity])/365
        self.strike_tot = df.Strike.to_numpy('float')
        self.Prices = df.Price.to_numpy('float')
        df_rates = pd.read_csv(
            '/Users/pierreranchet/Documents/Sauvegarde_PC/Dauphine/Cours/S2/Produits_Structures/Projet_Autocall/UK OIS spot curve.csv',
            sep=";")
        yield_maturities = df_rates['Maturities'].to_numpy('float')
        yields = df_rates['Rates'].to_numpy('float')
        curve_fit, status = calibrate_nss_ols(yield_maturities, yields)
        vfunc = np.vectorize(curve_fit)
        self.rate_tot = vfunc(self.maturity_tot)/100



    def prices_to_evaluate(self, x):
        v0, kappa, vbar, gamma, rho = [param for param in x]
        self.v0 = v0
        self.kappa = kappa
        self.vbar = vbar
        self.gamma = gamma
        self.rho = rho

        prices = []
        #for idx, val in enumerate(self.Prices):
            #self.maturity = self.maturity_tot[idx]
            #self.strike = self.strike_tot[idx]
            #self.rate = self.rate_tot[idx]
        self.vec_Heston_price = np.vectorize(self.GenerateHestonPaths)
        prices = self.vec_Heston_price(self.maturity_tot, self.strike_tot, self.rate_tot)
        #prices.append(self.GenerateHestonPaths(mat, K, rate))

        #prices = np.array(res)
        error = np.sum( (self.Prices-prices)**2 /len(self.Prices) )

        return error

    def calibrate(self):

        params = {"v0": {"x0": 0.1, "bd": [1e-3, 0.1]},
                  "kappa": {"x0": 3, "bd": [1e-3, 5]},
                  "vbar": {"x0": 0.05, "bd": [1e-3, 0.1]},
                  "gamma": {"x0": 0.3, "bd": [1e-2, 1]},
                  "rho": {"x0": -0.8, "bd": [-1, 0]}
                  }

        x0 = [param["x0"] for key, param in params.items()]
        bnds = [param["bd"] for key, param in params.items()]

        result = minimize(self.prices_to_evaluate, x0, tol=1e-3, method='SLSQP', options={'maxiter': 1e4}, bounds=bnds)
        print([param for param in result.x])
        v0, kappa, vbar, gamma, rho = [param for param in result.x]
        self.v0 = v0
        self.kappa = kappa
        self.vbar = vbar
        self.gamma = gamma
        self.rho = rho
        self.heston_prices = self.vec_Heston_price(self.maturity_tot, self.strike_tot, self.rate_tot)

        fig = go.Figure(data=[
            go.Mesh3d(x=self.maturity_tot, y=self.strike_tot, z=self.Prices, color='mediumblue',
                      opacity=0.55)])
        fig.add_scatter3d(x=self.maturity_tot, y=self.strike_tot, z=self.heston_prices,
                          mode='markers')
        fig.update_layout(
            title_text='Market Prices 𝑀𝑒𝑠ℎ vs Calibrated Heston Prices 𝑀𝑎𝑟𝑘𝑒𝑟𝑠',
            scene=dict(xaxis_title='TIME 𝑌𝑒𝑎𝑟𝑠',
                       yaxis_title='STRIKES 𝑃𝑡𝑠',
                       zaxis_title='INDEX OPTION PRICE 𝑃𝑡𝑠'),
            height=800,
            width=800
        )
        fig.show

        print('Hello')





test = Numerics(nb_simul=1000, nb_steps = 252, maturity=3.0, strike=100.0, rate=0.02, S_0=100.0, rho=-0.7, vbar=0.01, kappa =0.17 , gamma= 0.02, v0 = 0.15)
test.market_data()
test.calibrate()


