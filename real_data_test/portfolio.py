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

def backtest(tickers, start, end, lookback = 756, rebalance_freq = 63, risk_free_rate = 0.02, transaction_cost_rate = 0.0005):
    data = yf.download(tickers, start, end)
    closing_prices = data["Close"]
    df = pd.DataFrame(closing_prices)
    daily_returns = df.pct_change().dropna()
    actual_return = []
    weights_prev = np.zeros(len(tickers))

    for i in range(lookback, len(daily_returns) - rebalance_freq, rebalance_freq):
        mean_returns = daily_returns.iloc[i - lookback : i, :].mean()
        cov_matrix = daily_returns.iloc[i - lookback : i, :].cov()
        weights_new = max_sharpe_scipy(mean_returns, cov_matrix, risk_free_rate)

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

def plot_benchmark_comparison(tickers, start, end, lookback = 756, rebalance_freq = 63, risk_free_rate = 0.02):
    info = benchmark_comparison(tickers, start, end, lookback, rebalance_freq)
    my_strat_info = info[0]
    equal_weight_info = info[1]
    SPY_info = info[2]

    dates = [y for x,y in my_strat_info]
    my_strat_ret = [x for x,y in my_strat_info]
    equal_weights_ret = [x for x,y in equal_weight_info]
    SPY_backtest_ret = [x for x,y in SPY_info]

    my_strat_array = np.asarray(my_strat_ret)
    equal_weights_array = np.asarray(equal_weights_ret)
    spy_array = np.asarray(SPY_backtest_ret)

    my_strat_plot = np.cumprod(1 + my_strat_array)
    equal_weight_plot = np.cumprod(1 + equal_weights_array)
    SPY_plot = np.cumprod(1 + spy_array)

    plt.plot(dates, my_strat_plot, label="Strategy")
    plt.plot(dates, equal_weight_plot, label="Equal Weight")
    plt.plot(dates, SPY_plot, label="SPY")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
           "JPM", "BAC", "V", "MA",
           "JNJ", "PFE", "UNH", "MRK",
           "PG", "KO", "WMT", "COST",
           "HD", "NKE", "MCD",
           "XOM", "CVX",
           "GE", "BA", "CAT",
           "T", "VZ",
           "F", "GM",
           "DIS"]
    start = "2000-01-01"
    end = "2024-01-01"

    strategy_filtered, equal_weight_filtered, spy_filtered = benchmark_comparison(tickers, start, end)

    strategy_returns = [ret for ret, date in strategy_filtered]
    equal_weight_returns = [ret for ret, date in equal_weight_filtered]
    spy_returns = [ret for ret, date in spy_filtered]

    strat_return, strat_vol, strat_sharpe = backtest_summary(strategy_returns, risk_free_rate=0.02)
    ew_return, ew_vol, ew_sharpe = backtest_summary(equal_weight_returns, risk_free_rate=0.02)
    spy_return, spy_vol, spy_sharpe = backtest_summary(spy_returns, risk_free_rate=0.02)

    print(f"{'Strategy':<15}{'Return':>10}{'Vol':>10}{'Sharpe':>10}")
    print(f"{'My Strategy':<15}{strat_return:>10.4f}{strat_vol:>10.4f}{strat_sharpe:>10.4f}")
    print(f"{'Equal Weight':<15}{ew_return:>10.4f}{ew_vol:>10.4f}{ew_sharpe:>10.4f}")
    print(f"{'SPY':<15}{spy_return:>10.4f}{spy_vol:>10.4f}{spy_sharpe:>10.4f}")

    plot_benchmark_comparison(tickers, start, end)