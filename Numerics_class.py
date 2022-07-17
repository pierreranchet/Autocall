import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
import plotly.graph_objects as go
from Heston_class import Heston


class Numerics(Heston):
    def __init__(self, nb_simul, nb_steps, maturity, strike, rate, S_0, rho, vbar, kappa, gamma, v0):
        super().__init__(nb_simul, nb_steps, maturity, strike, rate, S_0, rho, vbar, kappa, gamma, v0)

    def market_data(self):
        ## Market datas from csv that ca be accessed from Autocall class with inheritance
        self.df_CDS_rate = pd.read_csv('CDS.csv', sep=";")
        self.df_CDS_rate.CDS = self.df_CDS_rate.CDS * 0.0001
        self.df_CDS_rate = self.df_CDS_rate.set_index(['time'], drop=True)

        df = pd.read_csv('FTSE_Prices.csv', sep=";")
        ## Spot fidex as of July 15th 22
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


    ## Objective function for the optimizer
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
        ## returns Mean-Squared Error
        return result / len(self.mkt_data)

    ## Facultative calibration
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

    ## Plot Heston prices vs Markers prices (matrices)
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