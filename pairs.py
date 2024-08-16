import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os
import numpy as np


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

    return data.fillna(0.00001)


def get_spread1(ticker1, ticker2, data, plot=False):
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
    # hedge_ratio = 1
    spread = data[ticker1] - hedge_ratio * data[ticker2]

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(spread, label=f'Spread: {ticker1} - {ticker2}')
        plt.axhline(spread.mean(), color='red', linestyle='--', label='Mean')
        plt.legend()
        plt.show()

    return spread


def get_spread2(ticker1, ticker2, data, plot=False):
    Y = np.log(data[ticker1])
    X = np.log(data[ticker2])
    X = sm.add_constant(X)  # Adds a constant column to X

    # Fit the OLS model
    model = sm.OLS(Y, X)
    results = model.fit()
    results.params
    # Extract beta and alpha from the model parameters
    alpha = results.params.iloc[0]
    beta = results.params.iloc[1]
    spread = Y - beta * X.iloc[:, 1] - alpha

    return spread


def get_zscore(spread):
    
    mean = np.mean(spread)
    std = np.std(spread)
    zscore = (spread - mean) / std
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
    tickers = data.columns
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            spread = get_spread2(tickers[i], tickers[j], data)
            adf = run_adf(spread, print_result=print_result, plot=plot)
            p_value = adf[1]

            if p_value < 0.05:
                valid_pairs.append((tickers[i], tickers[j]))

            if adf[0] < adf[4]['1%']:
                extra_valid_pairs.append((tickers[i], tickers[j]))
                # valid_pairs.remove((tickers[i], tickers[j]))

    return valid_pairs, extra_valid_pairs


def add_signals(data, valid_pairs):
    for pair in valid_pairs:
        print(pair)
        spread = get_spread2(pair[0], pair[1], data)
        zscore = get_zscore(spread)
        data[f'zscore {pair[0]}_{pair[1]}'] = zscore
        data[f'{pair[0]}_{pair[1]}_signal'] = 0
        data[f'{pair[0]}_{pair[1]}_signal'].loc[zscore > 1] = -1
        data[f'{pair[0]}_{pair[1]}_signal'].loc[zscore < -1] = 1
        zscore.plot()
        plt.title(f'zscore {pair[0]}_{pair[1]}')
        plt.axhline(1, color='red', linestyle='--')
        plt.axhline(-1, color='red', linestyle='--')
        plt.show()


if __name__ == '__main__':
    # tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BLNK', 'TSLA',
    #            'SPY', 'NIO', 'GOOG', 'BRK-B', 'BAC', 'JPM']
    # start = '2015-01-01'
    # end = '2023-01-01'
    # data = download_prices(tickers, start, end)
    # valid_pairs, xtra_valid_pairs = check_pairs(data)
    # print(f"Valid pairs: {valid_pairs}")
    # print(f"Xtra Valid pairs: {xtra_valid_pairs}")
    valid_pairs = [('BAC', 'JPM')]
    data = download_prices(['BAC', 'JPM'], '2015-01-01', '2023-01-01')
    add_signals(data, valid_pairs)
