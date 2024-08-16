import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os
import numpy as np
from analysis import cagr, sharpe_ratio

#implement dynamic rolling window
#implement stop loss and take profit
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
    return data


def run_ols(data, ticker1, ticker2):

    Y = np.log(data[ticker1])
    X = np.log(data[ticker2])
    X = sm.add_constant(X)  # Adds a constant column to X

    # Fit the OLS model
    model = sm.OLS(Y, X)
    results = model.fit()
    return results

def get_spread(ticker1, ticker2, data, plot=False):
    """
    Calculate the spread between two tickers using OLS regression on log-transformed prices.

    Inputs:
    ticker1: str, ticker symbol
    ticker2: str, ticker symbol
    data: pd.DataFrame, historical prices for the tickers
    plot: bool, if True, plot the spread

    Returns:
    spread: pd.Series, the spread between the two tickers
    """
    # Run OLS regression
    results = run_ols(data, ticker1, ticker2)

    # Log-transform prices
    Y = np.log(data[ticker1])
    X = np.log(data[ticker2])

    # Extract beta and alpha from the OLS model parameters
    alpha = results.params.iloc[0]  # Intercept
    beta = results.params.iloc[1]   # Slope

    # Calculate the spread
    spread = Y - beta * X - alpha

    # Plot the spread if requested
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(spread, label=f'Spread: {ticker1} - {ticker2}')
        plt.axhline(spread.mean(), color='red', linestyle='--', label='Mean')
        plt.legend()
        plt.show()

    return spread


def get_zscore(spread):
    """
    Calculate the z-score of the spread.

    Inputs:
    spread: pd.Series, the spread between two tickers

    Returns:
    zscore: pd.Series, the z-score of the spread
    """
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
    if the spread is stationary
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
    if adf_result[0] < adf_result[4]['5%']:
        return True

    return False


def check_pairs(data, window=250, print_result=False, plot=False):
    """
    Check for cointegration between pairs of tickers.

    Inputs:
    data: pd.DataFrame, historical prices for the tickers
    train_factor: float, percent of data to use as initial coint check

    Returns:
    valid_pairs: list of tuples, pairs of tickers that are cointegrated
    """
    data = data.head(window)
    print(data.tail())
    valid_pairs = []
    tickers = data.columns
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            spread = get_spread(tickers[i], tickers[j], data)
            coint = run_adf(spread, print_result=print_result, plot=plot)
            if coint:
                valid_pairs.append((tickers[i], tickers[j]))
    return valid_pairs


def add_signals(data, valid_pairs, window=250):
    #testing for a single pair for now
    for pair in valid_pairs:
        ticker1, ticker2 = pair
        colname = f"{ticker1}_{ticker2}"
        data[colname] = np.nan
        data[f'{colname}_zscore'] = np.nan
        # start our sliding window ending i
        # if window is coint then can generate signals for i+1
        for i in range(window, len(data)):
            win_data = data.iloc[i-window:i]
            spread = get_spread(ticker1, ticker2, win_data)
            #check if spread is cointegrated
            if run_adf(spread):
                zscore = get_zscore(spread)
                data.loc[data.index[i], f'{colname}_zscore'] = zscore.iloc[-1]
                # if zscore is greater than 1, buy ticker1 and sell ticker2
                if zscore.iloc[-1] > 1:
                    data.loc[data.index[i], colname] = -1
                # if zscore is less than -1, sell ticker1 and buy ticker2
                if zscore.iloc[-1] < -1:
                    data.loc[data.index[i], colname] = 1
                # if zscore is between -1 and 1, do nothing
                if -1 <= zscore.iloc[-1] <= 1:
                    data.loc[data.index[i], colname] = 0
        


def backtest(data, valid_pairs, window=250, stop_loss=None, take_profit=None):
    portfolio = pd.DataFrame(index=data.index)
    
    for pair in valid_pairs:
        ticker1, ticker2 = pair
        colname = f"{ticker1}_{ticker2}"
        data[colname].shift(1)
        
        holding1 = False
        holding2 = False
        money1 = 1
        money2 = 1
        shares1 = 0
        shares2 = 0
        
        for i in range(window, len(data)):
            # Handle ticker1 trading logic
            if data.loc[data.index[i], colname] == 1 and not holding1:
                shares1 = money1 / data[ticker1].iloc[i]
                money1 = 0
                holding1 = True
            elif data.loc[data.index[i], colname] == -1 and holding1:
                money1 = shares1 * data[ticker1].iloc[i]
                shares1 = 0
                holding1 = False

            # Stop loss and take profit logic for ticker1
            if holding1:
                entry_price1 = data[ticker1].iloc[i]  # Record the entry price
                if stop_loss and data[ticker1].iloc[i] <= \
                        (1 - stop_loss) * entry_price1:
                    money1 = shares1 * data[ticker1].iloc[i]
                    shares1 = 0
                    holding1 = False
                elif take_profit and data[ticker1].iloc[i] >= \
                        (1 + take_profit) * entry_price1:
                    money1 = shares1 * data[ticker1].iloc[i]
                    shares1 = 0
                    holding1 = False

            # Record the returns for ticker1
            if holding1:
                data.loc[data.index[i], f'{colname}_returns1'] = \
                        shares1 * data[ticker1].iloc[i]
            else:
                data.loc[data.index[i], f'{colname}_returns1'] = money1

            # Handle ticker2 trading logic (implementing the shorting logic)
            if data.loc[data.index[i], colname] == -1 and not holding2:
                shares2 = money2 / data[ticker2].iloc[i]
                money2 = 0
                holding2 = True
            elif data.loc[data.index[i], colname] == 1 and holding2:
                money2 = shares2 * data[ticker2].iloc[i]
                shares2 = 0
                holding2 = False

            # Stop loss and take profit logic for ticker2
            if holding2:
                entry_price2 = data[ticker2].iloc[i]  # Record the entry price
                if stop_loss and data[ticker2].iloc[i] <= \
                        (1 - stop_loss) * entry_price2:
                    money2 = shares2 * data[ticker2].iloc[i]
                    shares2 = 0
                    holding2 = False
                elif take_profit and data[ticker2].iloc[i] >= \
                        (1 + take_profit) * entry_price2:
                    money2 = shares2 * data[ticker2].iloc[i]
                    shares2 = 0
                    holding2 = False

            # Record the returns for ticker2
            if holding2:
                data.loc[data.index[i], f'{colname}_returns2'] = \
                    shares2 * data[ticker2].iloc[i]
            else:
                data.loc[data.index[i], f'{colname}_returns2'] = money2

        # Calculate the combined returns for the pair
        data[f'{colname}_returns'] = (data[f'{colname}_returns1'] + 
                                      data[f'{colname}_returns2']) / 2

        # Accumulate results in the portfolio DataFrame

        portfolio[colname] = data[f'{colname}_returns']

    # Aggregate total portfolio returns
    portfolio['Combined Return'] = portfolio.mean(axis=1)

    return portfolio


if __name__ == '__main__':
    tickers = ['NIO', 'BLNK', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 
               'AMZN', 'AMD', 'NVDA']
    start = '2014-01-01'
    end = '2024-01-01'
    data = download_prices(tickers, start, end)
    valid_pairs = check_pairs(data)
    print(f"Valid pairs: {valid_pairs}")
    add_signals(data, valid_pairs)
    portfolio = backtest(data, valid_pairs)
    print(portfolio)
    portfolio['Combined Return'].plot()
    plt.show()
    # get CAGR and Sharpe ratio
    print(f"CAGR: {cagr(portfolio['Combined Return'])}")
    print(f"Sharpe Ratio: {sharpe_ratio(portfolio['Combined Return'])}")
    
