import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import pandas as pd

# simulate one Geometric Brownian motion
# dS_{t} = mu * S_{t} * dt + sigma * S_{t} * dW_{t}
# time to maturity
TT = 1
mu = -0.2
sigma = 0.05
r = 0.05

# number of trading days
NUM_OF_DATE = 252
dt = 1/NUM_OF_DATE

# gaussian noise
noise = np.random.normal(0, 1, NUM_OF_DATE)
stock = np.zeros((NUM_OF_DATE))
stock[0] = 100

for i in range(1, NUM_OF_DATE):
    # stocks in P-dynamics
    # dW_{t} = sqrt(dt) * noise
    # drift mu
    stock[i] = stock[i - 1] + stock[i -  1] * (mu * dt + np.sqrt(dt) * sigma * noise[i])

# plt.plot(np.arange(0, NUM_OF_DATE, 1), stock)
# plt.show()

# simulate many paths Geometric Brownian Motion
num_paths = 10
stock_paths = np.zeros((num_paths, NUM_OF_DATE))
stock_paths[:, 0] = 100

for p in range(num_paths):
    noise = np.random.normal(0, 1, NUM_OF_DATE)

    for j in range(1, NUM_OF_DATE):
        # Stocks in Q-dynamics
        # drift r
        stock_paths[p, j] = stock_paths[p, j - 1] + stock_paths[p, j - 1] * (r * dt + sigma * noise[j])

    plt.plot(np.arange(0, NUM_OF_DATE, 1), stock_paths[p,:])

plt.show()

# compute call option price
# strike
K = 100

call_payoff = stock_paths[:, NUM_OF_DATE - 1] - K
# price by the Monte Carlo method
price_call_Monte_Carlo = math.exp(-r * TT) / num_paths * np.sum(call_payoff[call_payoff > 0])
print("Price call Monte Carlo: ", price_call_Monte_Carlo)

# BS formula
d1 = 1 / (sigma * math.sqrt(TT)) * (math.log(100/K) + (r + sigma ** 2 / 2) * TT)
d2 = d1 - sigma * math.sqrt(TT)

# price by Black - Scholes formula
sample = stats.norm(0, 1)
BS_call = 100 * sample.cdf(d1) - K * math.exp(-r * TT) * sample.cdf(d2)
print("Example Black Scholes call option pricing: ", BS_call)

# ---------------------------------
# real data
spx_2020 = pd.read_csv('SPX2020.csv')
spx_2020['ClosePrice'].plot(figsize=(10, 6), )
plt.xlabel('SPX close prices for the year 2020')
plt.show()

# number of trading days
num_trading_days = len(spx_2020['ClosePrice'])

# Estimate historical volatility
spx_prices = spx_2020['ClosePrice']
r = spx_prices / spx_prices.shift(1)
r = r[1:]
r = np.log(r)
r_average = np.mean(r)

print("Average returns: ", r_average)

# annual historical volatility
sigma_vol = np.sqrt(num_trading_days) * np.sqrt(1 / (num_trading_days - 1) * np.sum((r - r_average) ** 2))
print("Volatility: ", sigma_vol)

# option pricing example
spot = 3700
strike = 3700
dividend_rate = 0.0163
risk_free_rate = 0.0025
vol = 0.3475
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()

maturity_date = ql.Date().todaysDate()
calculation_date = maturity_date - ql.Period(1, ql.Years)
ql.Settings.instance().evaluationDate = calculation_date

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_rate, day_count))
volTS = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, calendar, vol, day_count))

european_exercise = ql.EuropeanExercise(maturity_date)
european_option = ql.EuropeanOption(payoff, european_exercise)

bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividendTS, riskFreeTS, volTS)

european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
option_price = european_option.NPV()
print('Option price: ', option_price)

# real data option pricing
option_jan_2021 = pd.read_csv('SPX2021.csv')
print(option_jan_2021.head())