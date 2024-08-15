import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os


def download_prices(tickers, start, end):
    """
    Download historical prices for a list of tickers.

    Inputs:
    tickers: list of str, ticker symbols
    start: str, start date
    end: str, end date

    Returns:
    prices: pd.DataFrame, historical prices for the tickers
    """
    tickers_str = '_'.join(tickers)
    file_path = f'data/{tickers_str}_{start}_{end}.csv'

    if not os.path.exists('data'):
        os.mkdir('data')

    if os.path.exists(file_path):
        print('Data already exists, loading from file.')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        print('Downloading data...')
        data = pd.DataFrame()

        for ticker in tickers:
            ticker_data = yf.download(ticker, start=start, end=end,
                                      interval='1d')
            data[ticker] = ticker_data['Adj Close']

        data.to_csv(file_path)

    return data.fillna(0)


def get_synthetic_spread(ticker1, ticker2, data, plot=False):
    """
    Calculate the spread between two tickers using OLS regression.

    Inputs:
    ticker1: str, ticker symbol
    ticker2: str, ticker symbol
    data: pd.DataFrame, historical prices for the tickers
    plot: bool, if True, plot the spread

    Returns:
    spread: pd.Series, the spread between the two tickers
    """
    train = pd.DataFrame()
    train['ticker1'] = data[ticker1]
    train['ticker2'] = data[ticker2]
    model = sm.OLS(train.ticker1, train.ticker2).fit()
    hedge_ratio = model.params.iloc[0]
    spread = data[ticker1] - hedge_ratio * data[ticker2]

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(spread, label=f'Spread: {ticker1} - {ticker2}')
        plt.axhline(spread.mean(), color='red', linestyle='--', label='Mean')
        plt.legend()
        plt.show()

    return spread


def get_zscore(spread, window=100):
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore


def run_adf(spread, print_result=False, plot=False):
    """
    Perform ADF test on the spread to check for stationarity

    Inputs:
    spread: pd.Series, the spread between two tickers
    print_result: bool, if True, print the ADF test results
    plot: bool, if True, plot the spread

    Returns:
    adf_result: tuple, ADF test results
    """
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(spread, label='Spread')
        plt.axhline(spread.mean(), color='red', linestyle='--', label='Mean')
        plt.legend()
        plt.show()

    adf_result = adfuller(spread, maxlag=1)

    if print_result:
        print(f't-stat value = {adf_result[0]}')
        print(f'p-value = {adf_result[1]}')
        print('Critical values:')
        for key, value in adf_result[4].items():
            print(f'   {key}: {value}')
        if adf_result[0] < adf_result[4]['1%']:
            print('Reject the null hypothesis at the 1% level (Stationary)')
        if adf_result[0] < adf_result[4]['5%']:
            print('Reject the null hypothesis at the 5% level (Stationary)')
        else:
            print('Failed to reject the null hypothesis (Non-Stationary)')

    return adf_result


def check_pairs(data, print_result=False, plot=False):
    """
    Check for cointegration between pairs of tickers.

    Inputs:
    data: pd.DataFrame, historical prices for the tickers

    Returns:
    valid_pairs: list of tuples, pairs of tickers that are cointegrated
    extra_valid_pairs: list of tuples, where t-stat < 1% critical value
    """
    valid_pairs = []
    extra_valid_pairs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            spread = get_synthetic_spread(tickers[i], tickers[j], data)
            adf = run_adf(spread, print_result=print_result, plot=plot)
            p_value = adf[1]

            if p_value < 0.05:
                valid_pairs.append((tickers[i], tickers[j]))

            if adf[0] < adf[4]['1%']:
                extra_valid_pairs.append((tickers[i], tickers[j]))
                valid_pairs.remove((tickers[i], tickers[j]))

    return valid_pairs, extra_valid_pairs


def generate_signals(z_score, entry_threshold=2.0, exit_threshold=0.5):
    signals = pd.Series(index=z_score.index)
    signals[z_score > entry_threshold] = -1  # Short spread
    signals[z_score < -entry_threshold] = 1  # Long spread
    signals[(z_score < exit_threshold) & (z_score > -exit_threshold)] = 0 
    print(signals)
    return signals


if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BLNK', 'TSLA',
               'SPY', 'NIO', 'GOOG', 'BRK-B', 'SPY']
    start = '2018-12-01'
    end = '2022-03-31'
    data = download_prices(tickers, start, end)
    valid_pairs, xtra_valid_pairs = check_pairs(data)
    print(f"Valid pairs: {valid_pairs}")
    print(f"Xtra Valid pairs: {xtra_valid_pairs}")
