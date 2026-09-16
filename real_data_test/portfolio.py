import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate = 0.02):
    weights = np.asarray(weights)
    mean_returns = np.asarray(mean_returns)
    cov_matrix = np.asarray(cov_matrix)

    port_return = np.dot(weights, mean_returns)
    port_var = weights.T @ cov_matrix @ weights
    port_vol = np.sqrt(port_var)

    daily_rf = (1 + risk_free_rate)**(1/252) - 1
    sharpe = (port_return - daily_rf)/port_vol

    return (port_return, port_vol, sharpe)

def objective_max_sharpe(weights, mean_returns, cov_matrix, risk_free_rate = 0.02):
    return -1 * portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def objective_min_vol(weights, mean_returns, cov_matrix, risk_free_rate = 0.02):
    return portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[1]

def objective_risk_parity(weights, cov_matrix):
    weights = np.asarray(weights)
    cov_matrix = np.asarray(cov_matrix)
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    risk_contribution = []
    for i in range(len(weights)):
        risk_contribution.append(weights[i] * (cov_matrix @ weights)[i] / port_vol)

    objective_sum = 0
    for i in range(len(weights)):
        objective_sum += (risk_contribution[i] - port_vol / len(weights)) ** 2

    return objective_sum

def constraint1(weights):
    sum = 0
    for i in range(len(weights)):
        sum += weights[i]
    return sum - 1

def constraint2(weights, mean_returns, cov_matrix, target_return, risk_free_rate = 0.02):
    return portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[0] - target_return

def max_sharpe_scipy(mean_returns, cov_matrix, max_bound = .25, risk_free_rate = 0.02):
    initial_guess = [1 / len(mean_returns)] * len(mean_returns)
    bounds = ((0, max_bound),) * len(mean_returns)

    cons1 = {'type': 'eq', 'fun': constraint1}
    cons = [cons1]
    result = minimize(objective_max_sharpe, initial_guess, method = 'SLSQP', constraints = cons, bounds = bounds, args = (mean_returns, cov_matrix, risk_free_rate))

    return result.x

def risk_parity_scipy(cov_matrix):
    length = len(cov_matrix)
    initial_guess = [1 / length] * length
    bounds = ((0, 1),) * length

    cons1 = {'type': 'eq', 'fun': constraint1}
    cons = [cons1]
    result = minimize(objective_risk_parity, initial_guess, method = 'SLSQP', constraints = cons, 
                      bounds = bounds, args = (cov_matrix, ), options = {'maxiter': 1000, 'ftol': 1e-12})

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

def get_universe_returns(tickers, start, end):
    data = yf.download(tickers, start, end)
    closing_prices = data["Close"]
    df = pd.DataFrame(closing_prices)
    daily_returns = df.pct_change().dropna()
    return daily_returns

def backtest(daily_returns, tickers, lookback = 756, rebalance_freq = 63, risk_free_rate = 0.02, transaction_cost_rate = 0.0005):
    actual_return = []
    weights_prev = np.zeros(len(tickers))

    for i in range(lookback, len(daily_returns) - rebalance_freq, rebalance_freq):
        mean_returns = daily_returns.iloc[i - lookback : i, :].mean()
        cov_matrix = daily_returns.iloc[i - lookback : i, :].cov()
        volatility_list = np.sqrt(np.diag(cov_matrix))

        individual_mean = [value for value in mean_returns]
        individual_std = [value for value in volatility_list]

        z_score_mean = (individual_mean - np.mean(individual_mean))/np.std(individual_mean)
        z_score_std = (individual_std - np.mean(individual_std))/np.std(individual_std)
        composite = (z_score_mean - z_score_std) / 2

        score_over_zero = []
        for index, value in enumerate(composite):
            if value > 0:
                score_over_zero.append((index, value))

        sorted(score_over_zero, key=lambda pair: pair[1])

        weights_new = np.zeros(len(tickers))
        for i in range(len(score_over_zero)):
            weights_new[i] = (2 * (i + 1) / (len(score_over_zero)*(len(score_over_zero) + 1)))

        turnover = np.sum(np.abs(weights_new - weights_prev))
        weights_prev = weights_new
        cost = turnover * transaction_cost_rate

        holding_slice = daily_returns.iloc[i + 1:i+rebalance_freq + 1, :]
        holding_dates = holding_slice.index

        quarter_returns = np.dot(holding_slice, weights_new)
        quarter_returns[0] = (1 + quarter_returns[0]) * (1 - cost) - 1

        paired = zip(quarter_returns, holding_dates)
        actual_return.extend(paired)

    return actual_return

def backtest_risk_parity(daily_returns, tickers, lookback = 756, rebalance_freq = 63, transaction_cost_rate = 0.0005):
    actual_return = []
    weights_prev = np.zeros(len(tickers))

    for i in range(lookback, len(daily_returns) - rebalance_freq, rebalance_freq):
        cov_matrix = daily_returns.iloc[i - lookback : i, :].cov()
        weights_new = risk_parity_scipy(cov_matrix)

        turnover = np.sum(np.abs(weights_new - weights_prev))
        weights_prev = weights_new
        cost = turnover * transaction_cost_rate

        holding_slice = daily_returns.iloc[i + 1:i+rebalance_freq + 1, :]
        holding_dates = holding_slice.index

        quarter_returns = np.dot(holding_slice, weights_new)
        quarter_returns[0] = (1 + quarter_returns[0]) * (1 - cost) - 1

        paired = zip(quarter_returns, holding_dates)
        actual_return.extend(paired)

    return actual_return


def one_dol_growth(tickers, start, end, lookback = 756, rebalance_freq = 63):
    backtest_info = benchmark_comparison(tickers, start, end, lookback, rebalance_freq)
    my_strat = backtest_info[0]
    equal_weight = backtest_info[1]
    spy = backtest_info[2]
    risk_parity = backtest_info[3]

    my_strat_cumul_plot = [x for x,y in my_strat]
    equal_weight_cumul_plot = [x for x,y in equal_weight]
    spy_cumul_plot = [x for x,y in spy]
    risk_parity_cumul_plot = [x for x,y in risk_parity]

    my_strat_cumul_plot = np.asarray(my_strat_cumul_plot)
    equal_weight_cumul_plot = np.asarray(equal_weight_cumul_plot)
    spy_cumul_plot = np.asarray(spy_cumul_plot)
    risk_parity_cumul_plot = np.asarray(risk_parity_cumul_plot)

    my_strat_cumul_return = np.cumprod(1 + my_strat_cumul_plot)
    equal_weight_cumul_return = np.cumprod(1 + equal_weight_cumul_plot)
    spy_cumul_return = np.cumprod(1 + spy_cumul_plot)
    risk_parity_cumul_return = np.cumprod(1 + risk_parity_cumul_plot)

    dates = [y for x,y in my_strat]

    plt.plot(dates, my_strat_cumul_return, color="#ff0000", linewidth=1.25, label = "My Strategy")
    plt.plot(dates, equal_weight_cumul_return, color="#00ff04", linewidth=1.25, label = "Equal Weight")
    plt.plot(dates, spy_cumul_return, color="#ffae00", linewidth=1.25, label = "S&P 500")
    plt.plot(dates, risk_parity_cumul_return, color="#0004ff", linewidth=1.25, label = "Risk Parity")
    plt.legend()
    plt.show()

def backtest_summary(returns, risk_free_rate = 0.02):
    returns = np.asarray(returns)
    cumul_return = np.cumprod(1 + returns)
    annual_vol = np.std(returns) * np.sqrt(252)
    annual_return = cumul_return[-1] ** (252/len(returns)) - 1
    sharpe = (annual_return - risk_free_rate) / annual_vol

    return annual_return, annual_vol, sharpe

def benchmark_comparison(tickers, start, end, lookback = 756, rebalance_freq = 63):
    daily_returns = get_universe_returns(tickers, start, end)
    my_strat = backtest(daily_returns, tickers, lookback, rebalance_freq)
    equal_weight = equal_weight_backtest(daily_returns, tickers, lookback)
    SPY_comparison = SPY_backtest(start, end)
    risk_parity = backtest_risk_parity(daily_returns, tickers, lookback, rebalance_freq)

    strategy_dates = [date for ret, date in my_strat]    
    equal_weight_dates = [date for ret, date in equal_weight]
    SPY_dates = [date for ret, date in SPY_comparison]
    risk_parity_dates = [date for ret, date in risk_parity]

    common_dates = set(strategy_dates) & set(equal_weight_dates) & set(SPY_dates) & set(risk_parity_dates)

    strategy_filtered = [pair for pair in my_strat if pair[1] in common_dates]
    equal_weight_filtered = [pair for pair in equal_weight if pair[1] in common_dates]
    spy_filtered = [pair for pair in SPY_comparison if pair[1] in common_dates]
    risk_parity_filtered = [pair for pair in risk_parity if pair[1] in common_dates]

    return strategy_filtered, equal_weight_filtered, spy_filtered, risk_parity_filtered

def equal_weight_backtest(daily_returns, tickers, lookback=756):
    weights = [1 / len(tickers)] * len(tickers)
    sliced = daily_returns.iloc[lookback:, :]

    paired = zip(np.dot(sliced, weights), sliced.index)
    return list(paired)

def SPY_backtest(start, end, lookback = 756):
    data = yf.download("SPY", start, end)
    closing_prices = data["Close"]
    df = pd.DataFrame(closing_prices)
    daily_returns = df.pct_change().dropna()

    paired = zip(daily_returns["SPY"].iloc[lookback:], daily_returns.index[lookback:])

    return list(paired)

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "AMZN", "NVDA",
           "JPM", "BAC", 
           "JNJ", "PFE", "UNH", "MRK",
           "PG", "KO", "WMT", "COST",
           "HD", "NKE", "MCD",
           "XOM", "CVX",
           "GE", "BA", "CAT",
           "T", "VZ",
           "F",
           "DIS"]
    start = "2000-01-01"
    end = "2024-01-01"

    print(one_dol_growth(tickers, start, end))