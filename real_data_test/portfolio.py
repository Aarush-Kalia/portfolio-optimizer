import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from data_loader import stock_data
import matplotlib.pyplot as plt

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

def backtest(tickers, start, end, lookback = 756, rebalance_freq = 63, risk_free_rate = 0.02):
    data = yf.download(tickers, start, end)
    closing_prices = data["Close"]
    df = pd.DataFrame(closing_prices)
    daily_returns = df.pct_change().dropna()
    actual_return = []

    for i in range(lookback, len(daily_returns) - rebalance_freq + 1, rebalance_freq):
        mean_returns = daily_returns.iloc[i - lookback : i, :].mean()
        cov_matrix = daily_returns.iloc[i - lookback : i, :].cov()

        weights = max_sharpe_scipy(mean_returns, cov_matrix, risk_free_rate)
        paired = zip(np.dot(daily_returns.iloc[i:i+rebalance_freq, :], weights), daily_returns.iloc[i:i+rebalance_freq, :].index)
        actual_return.extend(paired)

    return actual_return

def one_dol_growth(tickers, start, end, lookback = 756, rebalance_freq = 63, risk_free_rate = 0.02):
    backtest_info = backtest(tickers, start, end, lookback, rebalance_freq, risk_free_rate)

    cumul_plot = [x for x,y in backtest_info]
    cumul_plot = np.asarray(cumul_plot)
    cumul_return = np.cumprod(1 + cumul_plot)
    dates = [y for x,y in backtest_info]
    plt.plot(dates, cumul_return, color="#ff0000", linewidth=1.25)
    plt.show()

def backtest_summary(returns, risk_free_rate = 0.02):
    returns = np.asarray(returns)
    cumul_return = np.cumprod(1 + returns)
    annual_vol = np.std(returns) * np.sqrt(252)
    annual_return = cumul_return[-1] ** (252/len(returns)) - 1
    sharpe = (annual_return - risk_free_rate) / annual_vol

    return annual_return, annual_vol, sharpe

def benchmark_comparison(tickers, start, end, lookback = 756, rebalance_freq = 63):
    my_strat = backtest(tickers, start, end, lookback, rebalance_freq)
    equal_weight = equal_weight_backtest(tickers, start, end, lookback)
    SPY_comparison = SPY_backtest(start, end)

    strategy_dates = [date for ret, date in my_strat]
    equal_weight_dates = [date for ret, date in equal_weight]
    spy_dates = [date for ret, date in SPY_comparison]

    common_dates = set(strategy_dates) & set(equal_weight_dates) & set(spy_dates)

    strategy_filtered = [pair for pair in my_strat if pair[1] in common_dates]
    equal_weight_filtered = [pair for pair in equal_weight if pair[1] in common_dates]
    spy_filtered = [pair for pair in SPY_comparison if pair[1] in common_dates]

    return strategy_filtered, equal_weight_filtered, spy_filtered

def equal_weight_backtest(tickers, start, end, lookback=756):
    data = yf.download(tickers, start, end)
    closing_prices = data["Close"]
    df = pd.DataFrame(closing_prices)
    daily_returns = df.pct_change().dropna()

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
    #mean_returns = [.08, .12, .10]
    #cov_matrix = [[.0225, .01125, 0.015], [.01125, .0625, -0.01], [.015, -0.01, .04]]
    #print(weights)

    #min_vol = min_volatility_for_target(mean_returns, cov_matrix, target_return=.095)
    #print(min_vol)

    #weights = [0.625, 0.375, .0]
    #print(portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate = 0.02)[1])

    
    mean_returns, cov_matrix = stock_data(["CI", "UNH", "CVS"], "2000-01-01", "2024-01-01")
    weights = max_sharpe_scipy(mean_returns, cov_matrix, risk_free_rate=0.02)
    frontier = efficient_frontier(mean_returns, cov_matrix, num_points=20)

    #print(portfolio_stats(weights, mean_returns, cov_matrix)[2])

    #print(weights)
    #print(' vol    ret')
    #for vol, ret in frontier:
        #print(f"{vol:.3f} {ret:.3f}")

    #print(one_dol_growth(["CI", "UNH", "CVS"], "2000-01-01", "2024-01-01"))

    returns = [x for x, y in backtest(["CI", "UNH", "CVS"], "2000-01-01", "2024-01-01")]
    annual_return, annual_vol, sharpe = backtest_summary(returns, risk_free_rate=0.02)
    #print(f"Annualized Return: {annual_return:.4f}")
    #print(f"Annualized Volatility: {annual_vol:.4f}")
    #print(f"Sharpe Ratio: {sharpe:.4f}")

    compare = benchmark_comparison(["AAPL", "JNJ", "XOM"], "2000-01-01", "2024-01-01")
    print(len(compare[0]), len(compare[1]), len(compare[2]))
