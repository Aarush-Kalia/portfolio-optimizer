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

def efficient_frontier(mean_returns, cov_matrix, num_points = 20):
    mean_returns = np.asarray(mean_returns)
    target_values = np.linspace(min(mean_returns), max(mean_returns), num_points)
    vol_list = []
    return_list = []
    ret_vol_pair = []
    for value in target_values:
        min_vol_weight = min_volatility_for_target(mean_returns, cov_matrix, value)
        stats = portfolio_stats(min_vol_weight, mean_returns, cov_matrix, risk_free_rate=.02)
        vol_list.append(stats[1])
        return_list.append(stats[0])

    for i in range(len(vol_list)):
        ret_vol_pair.append((vol_list[i], return_list[i]))

    smallest_index = np.argmin(vol_list)
    return ret_vol_pair[smallest_index:]
    


if __name__ == "__main__":
    mean_returns = [.08, .12, .10]
    cov_matrix = [[.0225, .01125, 0.015], [.01125, .0625, -0.01], [.015, -0.01, .04]]
    #weights = max_sharpe_scipy(mean_returns, cov_matrix, risk_free_rate=0.02)
    #print(weights)

    #min_vol = min_volatility_for_target(mean_returns, cov_matrix, target_return=.095)
    #print(min_vol)

    #weights = [0.625, 0.375, .0]
    #print(portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate = 0.02)[1])

    frontier = efficient_frontier(mean_returns, cov_matrix, num_points=20)
    print('  vol     ret')
    for vol, ret in frontier:
        print(f"{vol:.5f} {ret:.5f}")