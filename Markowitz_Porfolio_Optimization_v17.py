# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 00:16:29 2024

@author: Can
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)['Adj Close']
    daily_returns = stock_data.pct_change()
    return daily_returns

def calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.sum(weights * mean_returns) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

def objective_function(weights, mean_returns, cov_matrix, risk_free_rate):
    _, _, sharpe_ratio = calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    return -sharpe_ratio

start_date = '2013-01-01'
end_date = '2023-01-01'
company_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'CMS', 'IBM', 'CSCO', 'D']

stock_data_dict = {symbol: get_stock_data(symbol, start_date, end_date) for symbol in company_symbols}
combined_data = pd.DataFrame(stock_data_dict).dropna()

mean_returns = combined_data.mean() * 252
cov_matrix = combined_data.cov() * 252

risk_free_rate = 0.0257

initial_weights = np.random.rand(len(company_symbols))
initial_weights /= np.sum(initial_weights)

bounds = tuple((0, 1) for asset in range(len(company_symbols)))

result = minimize(objective_function, initial_weights, args=(mean_returns, cov_matrix, risk_free_rate),
                  method='SLSQP', bounds=bounds)

optimal_weights = result['x']
optimal_weights[optimal_weights < 1e-10] = 0  # Consider very small weights as exactly zero
optimal_weights /= np.sum(optimal_weights)

optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio = calculate_portfolio_performance(optimal_weights, mean_returns, cov_matrix, risk_free_rate)

# Display Optimal Weights
print("Optimal Weights:")
for symbol, weight in zip(company_symbols, optimal_weights):
    print(f"{symbol}: {weight:.4f}")

# Monte Carlo simulation for random portfolios
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
risk_free_rate = 0.0257

for i in range(num_portfolios):
    weights = np.random.random(len(company_symbols))
    weights /= np.sum(weights)
    
    portfolio_return, portfolio_volatility, _ = calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    
    results[0,i] = portfolio_return
    results[1,i] = portfolio_volatility
    results[2,i] = (portfolio_return - risk_free_rate) / portfolio_volatility

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# Efficient Frontier with Random Portfolios
axes[0].scatter(results[1,:], results[0,:], c=results[2,:], marker='o', cmap='viridis', label='Random Portfolios')
axes[0].scatter(np.sqrt(np.diag(cov_matrix)), mean_returns, marker='o', color='b', label='Stocks')
axes[0].scatter(optimal_portfolio_volatility, optimal_portfolio_return, marker='*', color='r', s=500, label='Optimal Portfolio')

for i, symbol in enumerate(company_symbols):
    axes[0].annotate(symbol, (np.sqrt(cov_matrix.iloc[i, i]), mean_returns[i]), textcoords="offset points", xytext=(5, 5), ha='left')

axes[0].set_xlim(0, max(np.sqrt(np.diag(cov_matrix))) * 1.1)
axes[0].set_ylim(min(mean_returns) * 0.9, max(mean_returns) * 1.1)
axes[0].set_title('Efficient Frontier with Optimal Portfolio and Stocks')
axes[0].set_xlabel('Volatility')
axes[0].set_ylabel('Return')
axes[0].legend()

# Efficient Frontier with Random Portfolios (Zoomed)
axes[1].scatter(results[1,:], results[0,:], c=results[2,:], marker='o', cmap='viridis', label='Random Portfolios')
axes[1].scatter(optimal_portfolio_volatility, optimal_portfolio_return, marker='*', color='r', s=500, label='Optimal Portfolio')

axes[1].set_title('Efficient Frontier with Optimal Portfolio (Zoomed)')
axes[1].set_xlabel('Volatility')
axes[1].set_ylabel('Return')
axes[1].legend()

plt.tight_layout()
plt.show()

print(result)