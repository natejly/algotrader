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


def get_spread(ticker1, ticker2, data, plot=False):
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
    model = sm.OLS(data[ticker1], sm.add_constant(data[ticker2])).fit()
    hedge_ratio = model.params[ticker2]
    spread = data[ticker1] - hedge_ratio * data[ticker2]

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(spread, label=f'Spread: {ticker1} - {ticker2}')
        plt.axhline(spread.mean(), color='red', linestyle='--', label='Mean')
        plt.legend()
        plt.show()

    return spread


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


def check_pairs(tickers, start, end):
    """
    Check for cointegration between pairs of tickers.

    Inputs:
    tickers: list of str, ticker symbols
    start: str, start date
    end: str, end date

    Returns:
    valid_pairs: list of tuples, pairs of tickers that are cointegrated
    extra_valid_pairs: list of tuples, where t-stat < 1% critical value
    """
    data = download_prices(tickers, start, end)
    valid_pairs = []
    extra_valid_pairs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            spread = get_spread(tickers[i], tickers[j], data)
            adf = run_adf(spread, print_result=True, plot=False)
            p_value = adf[1]

            if p_value < 0.05:
                valid_pairs.append((tickers[i], tickers[j]))

            if adf[0] < adf[4]['1%']:
                extra_valid_pairs.append((tickers[i], tickers[j]))
                valid_pairs.remove((tickers[i], tickers[j]))

    return valid_pairs, extra_valid_pairs


if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BLNK', 'TSLA',
               'SPY', 'NIO', 'GOOG']
    start = '2023-01-01'
    end = '2024-01-01'
    valid_pairs = check_pairs(tickers, start, end)
    print(f"Valid pairs: {valid_pairs[0]}")
    print(f"Xtra Valid pairs: {valid_pairs[1]}")
