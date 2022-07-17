#spot = 7159.01 as of July 16
import numpy as np
import pandas as pd
from Numerics_class import Numerics
import matplotlib.pyplot as plt
import time

start = time.time()

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

    ## Compute DIP price
    def get_EQ_price(self):

        final_payoff = self.spots_matrix[:, -1].copy()
        for i in range(0,len(self.spots_matrix[:, -1])):
            if self.spots_matrix[i, -1] / self.S_0 > self.PDI_barrier:
                final_payoff[i] = 0
            else:
                final_payoff[i] = self.strike_AC * self.S_0 - self.spots_matrix[i, -1]
        eq_price = np.mean(final_payoff) * np.exp(-self.vfunc_rate(self.tenor)*self.tenor) / self.S_0
        return eq_price

    ## Compute conditional Autocall probabilities
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

## User can either specify Heston parameters with calibrate = False or calibrate with calibrate = True
product = Autocall(barrier_AC = 1, tenor = 6, coupon_pa = 0.0935, nb_simul = 10000, strike_AC = 1, PDI_barrier = 0.6, calibrate = False, kappa = 3.39, v0 = 0.1029, gamma = 0.2896, rho = -0.747, vbar = 0.0766)
#product = Autocall(barrier_AC = 1, tenor = 6, coupon_pa = 0.09, nb_simul = 2, strike_AC = 1, PDI_barrier = 0.6, calibrate = True)
Athena_Price = product.price()
print(Athena_Price)
end = time.time()
print(end - start, "seconds")