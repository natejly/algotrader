import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os
import numpy as np
from analysis import sharpe_ratio, cagr


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
    # Check if the data is already downloaded
    if os.path.exists(file_path):
        print('Data already exists, loading from file.')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        # if not download the data
        print('Downloading data...')
        data = pd.DataFrame()

        for ticker in tickers:
            ticker_data = yf.download(ticker, start=start, end=end,
                                      interval='1d')
            # only keep the adjusted close price
            data[ticker] = ticker_data['Adj Close']

        data.to_csv(file_path)
    return data


def run_ols(data, ticker1, ticker2):
    """
    Run Ordinary Least Squares (OLS) regression on two tickers.

    Inputs:
    data: pd.DataFrame, historical prices for the tickers
    ticker1: str, ticker symbol
    ticker2: str, ticker symbol

    Returns:
    results: OLS regression results
    """

    Y = np.log(data[ticker1])
    X = np.log(data[ticker2])
    # add constant b/c thats how life works
    X = sm.add_constant(X)

    # Fit the OLS model
    model = sm.OLS(Y, X)
    results = model.fit()
    return results


def get_spread(ticker1, ticker2, data):
    """
    Calculate the spread between two tickers using OLS regression
    on log-transformed prices.

    Inputs:
    ticker1: str, ticker symbol
    ticker2: str, ticker symbol
    data: pd.DataFrame, historical prices for the tickers
    plot: bool, if True, plot the spread

    Returns:
    spread: pd.Series, the spread between the two tickers
    """
    results = run_ols(data, ticker1, ticker2)

    # Log-transform prices
    Y = np.log(data[ticker1])
    X = np.log(data[ticker2])
    # intercept
    alpha = results.params.iloc[0]
    # slope
    beta = results.params.iloc[1]

    spread = Y - beta * X - alpha

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

    # Perform Augmented Dickey-Fuller test!
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
    # Using the 5% critical value
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
    # for all pairs of tickers
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            spread = get_spread(tickers[i], tickers[j], data)
            coint = run_adf(spread, print_result=print_result, plot=plot)
            # if cointegrated add to valid pairs
            if coint:
                valid_pairs.append((tickers[i], tickers[j]))
    return valid_pairs


def add_signals(data, valid_pairs, window=250):
    """
    Adds trading signals to the data

    Inputs:
    data: pd.DataFrame, historical prices for the tickers
    valid_pairs: list of tuples, pairs of tickers that are cointegrated
    window: int, lookback window for the strategy

    Returns:
    data: pd.DataFrame, the data with trading signals
    """
    # Iterate through each valid cointegrated pair
    for pair in valid_pairs:
        ticker1, ticker2 = pair
        colname = f"{ticker1}_{ticker2}"

        # Initialize columns for trading signals and other metrics
        data[colname] = 0  # Default trading signal
        data[f'{colname}_zscore'] = np.nan
        data[f'{colname}_z_upper'] = np.nan
        data[f'{colname}_z_lower'] = np.nan
        data[f'{colname}_fitted'] = np.nan
        data[f'{colname}_residual'] = np.nan

        prev_status = False  # Track the previous cointegration status

        # Loop through the data with a sliding window approach
        for i in range(window, len(data)):
            # historical data for the window
            win_data = data.iloc[i-window:i]

            # Calculate spread and run OLS regression
            spread = get_spread(ticker1, ticker2, win_data)
            results = run_ols(win_data, ticker1, ticker2)

            # Perform ADF test to check if the spread is cointegrated
            coint = run_adf(spread)

            # If previously cointegrated but no longer cointegrated
            if prev_status and not coint:
                data.loc[data.index[i], colname] = 0  # Set signal to 0
                data.loc[data.index[i], f'{colname}_zscore'] = np.nan
                data.loc[data.index[i], f'{colname}_fitted'] = np.nan
                data.loc[data.index[i], f'{colname}_residual'] = np.nan

            # If previously not cointegrated but now cointegrated
            if not prev_status and coint:
                alpha = results.params.iloc[0]  # OLS intercept
                beta = results.params.iloc[1]   # OLS slope

                # Compute fitted values and residuals
                fitted = alpha + beta * np.log(data[ticker2].iloc[i:])
                residual = np.log(data[ticker1].iloc[i:]) - fitted

                # Calculate z-score for the residuals
                zscore = (residual - residual.mean()) / residual.std()
                data.loc[data.index[i:], f'{colname}_zscore'] = zscore
                data.loc[data.index[i:], f'{colname}_z_upper'] = \
                    zscore.mean() + residual.std()
                data.loc[data.index[i:], f'{colname}_z_lower'] = \
                    zscore.mean() - residual.std()

            # Generate trading signals based on z-score
            if coint:
                zscore = data.loc[data.index[i], f'{colname}_zscore']
                if zscore > data.loc[data.index[i], f'{colname}_z_upper']:
                    data.loc[data.index[i], colname] = -1  # Signal to sell
                elif zscore < data.loc[data.index[i], f'{colname}_z_lower']:
                    data.loc[data.index[i], colname] = 1   # Signal to buy
                else:
                    data.loc[data.index[i], colname] = 0   # No signal

            # Update previous cointegration status
            prev_status = coint

    return data


def backtest(data, valid_pairs, window=252):
    """"
    Function to backtest the pairs trading strategy

    Inputs:
    data: pd.DataFrame, historical prices for the tickers
    valid_pairs: list of tuples, pairs of tickers that are cointegrated
    window: int, lookback window for the strategy

    Returns:
    portfolio: pd.DataFrame, the portfolio returns
    """
    portfolio = pd.DataFrame()
    for pair in valid_pairs:
        ticker1, ticker2 = pair
        colname = f"{ticker1}_{ticker2}"
        data[colname].shift(1)
        holding = False
        money = 1
        shares = 0
        # for ticker2
        holding2 = False
        money2 = 1
        shares2 = 0
        for i in range(window, len(data)):
            if data.loc[data.index[i], colname] == 1 and not holding:
                shares = money / data[ticker1].iloc[i]
                money = 0
                holding = True
            elif data.loc[data.index[i], colname] == -1 and holding:
                money = shares * data[ticker1].iloc[i]
                shares = 0
                holding = False

            if holding:
                data.loc[data.index[i], f'{colname}_returns1'] = \
                    data[ticker1].iloc[i] * shares
            else:
                data.loc[data.index[i], f'{colname}_returns1'] = money

            # for ticker 2 which is the reverse of ticker 1
            if data.loc[data.index[i], colname] == -1 and not holding2:
                shares2 = money2 / data[ticker2].iloc[i]
                money2 = 0
                holding2 = True
            elif data.loc[data.index[i], colname] == 1 and holding2:
                money2 = shares2 * data[ticker2].iloc[i]
                shares2 = 0
                holding2 = False

            if holding2:
                data.loc[data.index[i], f'{colname}_returns2'] = \
                    data[ticker2].iloc[i] * shares2
            else:
                data.loc[data.index[i], f'{colname}_returns2'] = money2

        data[f'{colname}_returns'] = \
            (data[f'{colname}_returns1'] + data[f'{colname}_returns2']) / 2

        # plot returns

        portfolio[f'{colname}_returns'] = data[f'{colname}_returns']
    portfolio.dropna(inplace=True)
    portfolio['Combined Return'] = portfolio.sum(axis=1) / len(valid_pairs)
    portfolio['Percent Change'] = portfolio['Combined Return'].pct_change()
    return portfolio


def main():
    """
    Main function to run the pairs trading strategy
    """
    # FAANG and other tech stocks + SPY and QQQ
    tickers = ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL',
               'MSFT', 'TSLA', 'NVDA', 'AMD', 'SPY', 'QQQ']
    # 5 years of data from today for 20/80 train/test split
    # Note: accidentally started at 2018 once and almost fried my computer
    # so be careful but also got a sharpe of 1.5 w/ 30 cagr
    start = '2019-8-16'
    end = '2024-08-16'
    data = download_prices(tickers, start, end)
    # high risk of overfiltering
    # could run all pairs but thats a lot of computing
    # so we will keep to cointegrated pairs in the training data
    valid_pairs = check_pairs(data)
    print(f"Valid pairs: {valid_pairs}")
    add_signals(data, valid_pairs)
    print(data.head)
    portfolio = backtest(data, valid_pairs)
    print(portfolio)
    portfolio.dropna(inplace=True)
    portfolio['Combined Return'].plot()
    plt.title('Combined Strategy Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.grid()
    # put drawdown and other stuff here
    print(f"Strategy Return: {portfolio['Combined Return'].iloc[-1]}")
    print(f"CAGR: {cagr(portfolio['Combined Return'])}")
    print(f"Sharpe Ratio: {sharpe_ratio(portfolio['Percent Change'])}")
    plt.show()


if __name__ == '__main__':
    main()
