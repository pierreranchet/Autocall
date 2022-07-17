#spot = 7159.01 as of July 16
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize, fmin

from nelson_siegel_svensson.calibrate import calibrate_ns_ols, calibrate_nss_ols
import plotly.graph_objects as go


cnt_optimizer = 0

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
        paths_time = self.get_spots_Heston(mat, K, rate)["S"]
        return np.exp(-self.rate* self.maturity) * np.mean(np.maximum(self.strike - paths_time[:,-1], 0))


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
            V[:, i + 1] = V[:, i] + self.kappa * (self.vbar - V[:, i]) * dt + self.gamma * np.sqrt(V[:, i]) * (
                        W1[:, i + 1] - W1[:, i])
            V[:, i + 1] = np.maximum(V[:, i + 1], 0.0)

            X[:, i + 1] = X[:, i] + (self.rate - 0.5 * V[:, i]) * dt + np.sqrt(V[:, i]) * (W2[:, i + 1] - W2[:, i])
            time[i + 1] = time[i] + dt

        # Compute exponent
        self.spots_MC = np.exp(X)
        self.paths = {"time": time, "S": self.spots_MC}
        return self.paths


class Numerics(Heston):
    def __init__(self, nb_simul, nb_steps, maturity, strike, rate, S_0, rho, vbar, kappa, gamma, v0):
        super().__init__(nb_simul, nb_steps, maturity, strike, rate, S_0, rho, vbar, kappa, gamma, v0)

    def market_data(self):
        ## Market datas from csv that ca be accessed from Autocall class with inheritance
        self.df_CDS_rate = pd.read_csv('CDS.csv', sep=";")
        self.df_CDS_rate.CDS = self.df_CDS_rate.CDS * 0.0001
        self.df_CDS_rate = self.df_CDS_rate.set_index(['time'], drop=True)

        df = pd.read_csv('FTSE_Prices.csv', sep=";")

        self.S_0 = 7159.01
        self.maturity_tot = np.array([(datetime.strptime(i, '%m/%d/%Y') - datetime.today()).days for i in df.Maturity])/365
        self.strike_tot = df.Strike.to_numpy('float')
        self.Prices = df.Price.to_numpy('float')
        df_rates = pd.read_csv('UK OIS spot curve.csv',sep=";")
        yield_maturities = df_rates['Maturities'].to_numpy('float')
        yields = df_rates['Rates'].to_numpy('float')/100
        curve_fit, status = calibrate_ns_ols(yield_maturities, yields, tau0=1.0)
        self.vfunc_rate = np.vectorize(curve_fit)
        self.rate_tot = self.vfunc_rate(self.maturity_tot)
        self.mkt_data = np.vstack((self.maturity_tot, self.strike_tot, self.rate_tot, self.Prices)).T



    def prices_to_evaluate(self, x):
        global cnt_optimizer
        cnt_optimizer += 1
        print(cnt_optimizer)

        v0, kappa, vbar, gamma, rho = [param for param in x]
        self.v0 = v0
        self.kappa = kappa
        self.vbar = vbar
        self.gamma = gamma
        self.rho = rho

        result = 0.0
        for mkt in self.mkt_data:
            mat, k, r, price_off = mkt
            eval = self.GenerateHestonPaths(mat,k,r)
            result += (eval - price_off) ** 2

        return result / len(self.mkt_data)


    def calibrate_func(self):

        params = {"v0": {"x0": 0.1029, "bd": [1e-3, 1]},
                  "kappa": {"x0": 3.39, "bd": [1e-3, 10]},
                  "vbar": {"x0": 0.0766, "bd": [1e-3, 0.8]},
                  "gamma": {"x0": 0.2896, "bd": [1e-2, 2]},
                  "rho": {"x0": -0.747, "bd": [-1, 1]}
                  }

        x0 = np.array([param["x0"] for key, param in params.items()])
        bnds = [param["bd"] for key, param in params.items()]
        result = minimize(self.prices_to_evaluate, x0, tol=50, method='Nelder-Mead', options={'maxiter': 1}, bounds=bnds)
        self.v0, self.kappa, self.vbar, self.gamma, self.rho = result.x
        ## To check the value of the final loss function after calibration
        #error = self.prices_to_evaluate(x0)

        return x0

    def plot_prices_surface(self):
        print('Computing prices matrix with Heston')
        self.vec_Heston_price = np.vectorize(self.GenerateHestonPaths)
        self.Heston_Prices = self.vec_Heston_price(self.maturity_tot, self.strike_tot, self.rate_tot)

        fig = go.Figure(data=[
            go.Mesh3d(x=self.maturity_tot, y=self.strike_tot, z=self.Prices, color='mediumblue',
                      opacity=0.55), go.Mesh3d(x=self.maturity_tot, y=self.strike_tot, z=self.Heston_Prices, color='red',
                      opacity=0.55)])

        fig.update_layout(
            title_text='Market Prices Mesh vs Calibrated Heston Prices Markers',
            scene=dict(xaxis_title='TIME 𝑌𝑒𝑎𝑟𝑠',
                       yaxis_title='STRIKES 𝑃𝑡𝑠',
                       zaxis_title='INDEX OPTION PRICE 𝑃𝑡𝑠'),
            height=800,
            width=800
        )
        fig.show()


class Autocall(Numerics):
    def __init__(self, barrier_AC, tenor, coupon_pa, nb_simul, strike_AC, PDI_barrier, calibrate, kappa = None, v0 = None, gamma = None, rho = None, vbar = None):
        super().__init__(nb_simul = nb_simul, nb_steps = 1000, maturity = tenor, strike = strike_AC, rate = 0.02, S_0 = 7159.01, rho = rho, vbar = vbar, kappa = kappa, gamma = gamma, v0 = v0)
        self.barrier_AC = barrier_AC
        self.tenor = tenor
        self.coupon_pa = coupon_pa
        self.nb_simul = nb_simul
        self.strike_AC = strike_AC
        self.PDI_barrier = PDI_barrier
        self.calibrate = calibrate
        if not calibrate:
            self.kappa = kappa
            self.rho = rho
            self.gamma = gamma
            self.v0 = v0
            self.rho = rho


    def get_EQ_price(self):

        final_payoff = self.spots_matrix[:, -1].copy()
        for i in range(0,len(self.spots_matrix[:, -1])):
            if self.spots_matrix[i, -1] / self.S_0 > self.PDI_barrier:
                final_payoff[i] = 0
            else:
                final_payoff[i] = self.strike_AC * self.S_0 - self.spots_matrix[i, -1]
        eq_price = np.mean(final_payoff) * np.exp(-self.vfunc_rate(self.tenor)*self.tenor) / self.S_0
        return eq_price

    def AC_probabilities(self):
        self.liste_t = [2,3,4,5,6]
        self.list_obs = [round((i/self.tenor) * self.nb_steps) for i in self.liste_t]
        times = self.get_spots_Heston(rho=self.rho, v0=self.v0, vbar=self.vbar, gamma=self.gamma)["time"]
        self.discrete_spots = self.spots_matrix[:,self.list_obs]
        probas = []
        probas.append(np.where((self.discrete_spots[:,0] > self.S_0) , self.discrete_spots[:,0] > self.S_0, 0).sum() / self.nb_simul)
        probas.append(np.where((self.discrete_spots[:, 0] < self.S_0),self.discrete_spots[:, 1] > self.S_0, 0).sum() / self.nb_simul)
        probas.append(np.where((self.discrete_spots[:, 0] < self.S_0) & (self.discrete_spots[:, 1] < self.S_0), self.discrete_spots[:, 2] > self.S_0, 0).sum() /  self.nb_simul)
        probas.append(np.where((self.discrete_spots[:, 0] < self.S_0) & (self.discrete_spots[:, 1] < self.S_0) & (self.discrete_spots[:, 2] < self.S_0), self.discrete_spots[:, 3] > self.S_0, 0).sum() / self.nb_simul)
        probas.append(1-sum(probas))
        return probas

    def price_coupon_AC(self):
        df_temp = pd.DataFrame(self.discrete_spots, columns=self.liste_t)
        payoff_temp = df_temp.copy() * 0
        cnt = 0
        for i in range(len(df_temp.index)):
            for j in self.liste_t:
                if df_temp.loc[i, j] >  self.barrier_AC * self.S_0:
                    payoff_temp.loc[i, j] = self.coupon_pa * j * np.exp(-self.vfunc_rate(j) * j)
                    break
        return payoff_temp.sum(axis=1).sum()/self.nb_simul


    def price(self):
        self.market_data()
        if self.calibrate:
            self.calibrate_func()

        #self.plot_prices_surface()

        self.spots_matrix = self.get_spots_Heston(rho=self.rho, v0=self.v0, vbar=self.vbar, gamma=self.gamma)["S"]
        DIP_price = self.get_EQ_price()
        AC_probas = self.AC_probabilities()
        funding = ((self.df_CDS_rate.CDS + self.df_CDS_rate.sofr) * self.df_CDS_rate.index)[self.liste_t].dot(AC_probas)
        price_coupon = self.price_coupon_AC()
        final_price = 1 - DIP_price - funding + price_coupon
        return final_price

product = Autocall(barrier_AC = 1, tenor = 6, coupon_pa = 0.0935, nb_simul = 1000, strike_AC = 1, PDI_barrier = 0.6, calibrate = False, kappa = 3.39, v0 = 0.1029, gamma = 0.2896, rho = -0.747, vbar = 0.0766)
#product = Autocall(barrier_AC = 1, tenor = 6, coupon_pa = 0.09, nb_simul = 2, strike_AC = 1, PDI_barrier = 0.6, calibrate = True)
product.price()



