import numpy as np
import scipy.stats as si
import math

S = 100
K = 100
r = 0.05
sigma = 0.2
T = 1
C0 = 8.02
P0 = 11.98

def d1(S, K, r, sigma, T):
    return (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
def d2(S, K, r, sigma, T):
    return (np.log(S/K)+(r-sigma**2/2)*T)/(sigma*np.sqrt(T))

def call_price(S, K, r, sigma, T):
    return S*si.norm.cdf(d1(S, K, r, sigma, T), 0.0, 1.0)-K*np.exp(-r*T)*si.norm.cdf(d2(S, K, r, sigma, T), 0.0, 1.0)
def put_price(S, K, r, sigma, T):
    return K*np.exp(-r*T)*si.norm.cdf(-d2(S, K, r, sigma, T), 0.0, 1.0)-S*si.norm.cdf(-d1(S, K, r, sigma, T), 0.0, 1.0)


def put_vega(S, K, r, sigma, T):
    return S*np.sqrt(T)*si.norm.pdf(d1(S, K, r, sigma, T), 0.0, 1.0)
def call_vega(S, K, r, sigma, T):
    return S*np.sqrt(T)*si.norm.pdf(d1(S, K, r, sigma, T), 0.0, 1.0)


def call_imp_vol(S, K, r, T, C0, sigma_est, it=100):
    for i in range(it):
        sigma_est -= ((call_price(S, K, r, sigma_est, T)-C0)/call_vega(S, K, r, sigma_est, T))
    return sigma_est

def put_imp_vol(S, K, r, T, P0, sigma_est, it=100):
    for i in range(it):
        sigma_est -= ((put_price(S, K, r, sigma_est, T)-P0)/put_vega(S, K, r, sigma_est, T))
    return sigma_est


print('Call Price: ', call_price(S, K, r, sigma, T))
print('Put Price: ', put_price(S, K, r, sigma, T))
print('Call Vega: ', call_vega(S, K, r, sigma, T))
print('Put Vega: ', put_vega(S, K, r, sigma, T))

print('Implied Volatility of Call Option: ', str(round(call_imp_vol(S, K, r, T, C0, sigma)*100,2))+ '%')
print('Implied Volatility of Put Option: ', str(round(put_imp_vol(S, K, r, T, P0, sigma)*100,2))+ '%')
