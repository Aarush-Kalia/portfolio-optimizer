import numpy as np
from scipy.optimize import minimize

def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate = 0.02):
    weights = np.asarray(weights)
    mean_returns = np.asarray(mean_returns)
    cov_matrix = np.asarray(cov_matrix)

    port_return = np.dot(weights, mean_returns)
    port_var = weights.T @ cov_matrix @ weights
    port_vol = np.sqrt(port_var)

    sharpe = (port_return - risk_free_rate)/port_vol

    return (port_return, port_vol, sharpe)

def objective_max_sharpe(weights, mean_returns, cov_matrix, risk_free_rate = 0.02):
    return -1 * portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def objective_min_vol(weights, mean_returns, cov_matrix, risk_free_rate = 0.02):
    return portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[1]

def constraint1(weights):
    sum = 0
    for i in range(len(weights)):
        sum += weights[i]
    return sum - 1

def constraint2(weights, mean_returns, cov_matrix, target_return, risk_free_rate = 0.02):
    return portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[0] - target_return

def max_sharpe_scipy(mean_returns, cov_matrix, risk_free_rate = 0.02):
    initial_guess = [1 / len(mean_returns)] * len(mean_returns)
    bounds = ((0, 1),) * len(mean_returns)

    cons1 = {'type': 'eq', 'fun': constraint1}
    cons = [cons1]
    result = minimize(objective_max_sharpe, initial_guess, method = 'SLSQP', constraints = cons, bounds = bounds, args = (mean_returns, cov_matrix, risk_free_rate))

    return result.x

def min_volatility_for_target(mean_returns, cov_matrix, target_return):
    initial_guess = [1 / len(mean_returns)] * len(mean_returns)
    bounds = ((0, 1),) * len(mean_returns)

    cons1 = {'type': 'eq', 'fun': constraint1}
    cons2 = {'type': 'eq', 'fun': constraint2, 'args': (mean_returns, cov_matrix, target_return)}
    cons = [cons1, cons2]
    result = minimize(objective_min_vol, initial_guess, method = 'SLSQP', constraints = cons, bounds = bounds, args = (mean_returns, cov_matrix))
    return result.x

if __name__ == "__main__":
    mean_returns = [.08, .12]
    cov_matrix = [[.0225, .01125], [.01125, .0625]]
    weights = max_sharpe_scipy(mean_returns, cov_matrix, risk_free_rate=0.02)
    #print(weights)

    min_vol = min_volatility_for_target(mean_returns, cov_matrix, target_return=.09)
    #print(min_vol)

    weights = [0.75, 0.25]
    print(portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate = 0.02)[0])