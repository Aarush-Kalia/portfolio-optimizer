import pandas as pd
import yfinance as yf
import numpy as np

def stock_data(stocks_list, start, end):
    data = yf.download(stocks_list, start, end)
    closing_prices = data["Close"]
    df = pd.DataFrame(closing_prices)

    daily_returns = df.pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    return mean_returns, cov_matrix