import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nelson_siegel_svensson.calibrate import calibrate_ns_ols, calibrate_nss_ols
from datetime import datetime
from scipy.optimize import minimize, fmin




def GeneratePathsHestonEuler(NoOfPaths, NoOfSteps, T, r, S_0, kappa, gamma, rho, vbar, v0, strike):

    Z1 = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    Z2 = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W1 = np.zeros([NoOfPaths, NoOfSteps + 1])
    W2 = np.zeros([NoOfPaths, NoOfSteps + 1])
    V = np.zeros([NoOfPaths, NoOfSteps + 1])
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    V[:, 0] = v0
    X[:, 0] = np.log(S_0)

    time = np.zeros([NoOfSteps + 1])

    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z1[:, i] = (Z1[:, i] - np.mean(Z1[:, i])) / np.std(Z1[:, i])
            Z2[:, i] = (Z2[:, i] - np.mean(Z2[:, i])) / np.std(Z2[:, i])
        Z2[:, i] = rho * Z1[:, i] + np.sqrt(1.0 - rho ** 2) * Z2[:, i]

        W1[:, i + 1] = W1[:, i] + np.power(dt, 0.5) * Z1[:, i]
        W2[:, i + 1] = W2[:, i] + np.power(dt, 0.5) * Z2[:, i]

        # Truncated boundary condition
        V[:, i + 1] = V[:, i] + kappa * (vbar - V[:, i]) * dt + gamma * np.sqrt(V[:, i]) * (W1[:, i + 1] - W1[:, i])
        V[:, i + 1] = np.maximum(V[:, i + 1], 0.0)

        X[:, i + 1] = X[:, i] + (r - 0.5 * V[:, i]) * dt + np.sqrt(V[:, i]) * (W2[:, i + 1] - W2[:, i])
        time[i + 1] = time[i] + dt

    # Compute exponent
    S = np.exp(X)
    price = np.exp(r * T) * np.mean(np.maximum(strike - S[:, -1], 0))
    return price

def evaluate(x):
    kappa, gamma, rho, vbar, v0 = x
    market_data = pd.read_csv(
        '/Users/pierreranchet/Documents/Sauvegarde_PC/Dauphine/Cours/S2/Produits_Structures/Projet_Autocall/FTSE_Prices.csv',
        sep=";")
    market_data.Maturity = np.array(
        [(datetime.strptime(i, '%m/%d/%Y') - datetime.today()).days for i in market_data.Maturity]) / 365
    df_rates = pd.read_csv(
        '/Users/pierreranchet/Documents/Sauvegarde_PC/Dauphine/Cours/S2/Produits_Structures/Projet_Autocall/UK OIS spot curve.csv',
        sep=";")
    yield_maturities = df_rates['Maturities'].to_numpy('float')
    yields = df_rates['Rates'].to_numpy('float') / 100
    curve_fit, status = calibrate_ns_ols(yield_maturities, yields, tau0=1.0)
    market_data['Rates'] = curve_fit(market_data['Maturity'])


    error = 0.0
    for data in market_data.values:

        k, price, mat, r = data
        price_heston = GeneratePathsHestonEuler(NoOfPaths = 1000, NoOfSteps = 1000, T = mat, r = r, S_0 = 7323.41, kappa = kappa, gamma = gamma, rho = rho, vbar = vbar, v0 = v0, strike = k)
        error = (price_heston - price)**2

    return error


def calibrate():
    #x = np.array([3.0, 0.25, -0.7, 0.15, 0.1])
    #error = evaluate(x)

    params = {"kappa": {"x0": 3.0, "bd": [1e-3, 10]},
              "gamma": {"x0": 0.25, "bd": [1e-2, 2]},
              "rho": {"x0": -0.7, "bd": [-1, 1]},
              "vbar": {"x0": 0.15, "bd": [1e-3, 0.8]},
              "v0": {"x0": 0.1, "bd": [1e-3, 1]}
              }

    x0 = np.array([param["x0"] for key, param in params.items()])
    result = minimize(evaluate, x0, tol=1e-3, method='Nelder-Mead', options={'maxiter': 1e4})
    result2 = fmin(evaluate, x0, maxiter = 50)

    print('HH')




calibrate()


S = GeneratePathsHestonEuler(NoOfPaths = 1000, NoOfSteps = 1000, T = 6, r = 0.02, S_0 = 7323.41, kappa = 3.0, gamma = 0.25, rho = -0.7, vbar = 0.12, v0 = 0.1)['S']
price = np.exp(0.02*6) * np.mean(np.maximum(7323.41 - S[:,-1], 0))

print('Hello')


