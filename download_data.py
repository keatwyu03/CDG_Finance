"""
Download historical stock data and create Stocks_logret.csv
Run once before main.py: python download_data.py
Requires: pip install yfinance
"""
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Install yfinance first:  pip install yfinance")

TICKERS = ["AAPL", "AMZN", "JPM", "TSLA"]
START = "2010-06-30"
END = "2023-04-12"

print(f"Downloading {TICKERS} from {START} to {END} ...")
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=True)

# Use adjusted close prices
prices = raw["Close"][TICKERS].dropna()

# Log returns
log_ret = np.log(prices / prices.shift(1)).dropna()

# Add weekday column (0=Mon … 4=Fri)
log_ret["weekday"] = log_ret.index.dayofweek

log_ret.index.name = "Date"
log_ret = log_ret.reset_index()

out = "Stocks_logret.csv"
log_ret.to_csv(out, index=False)
print(f"Saved {len(log_ret)} rows to {out}")
print(log_ret.head())
