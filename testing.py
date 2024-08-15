'''
Simple stragegy as proof of concept
1. Download data from Yahoo Finance
2. Add Bollinger Bands to the data
3. Add a simple trading signal based on the Bollinger Bands
    if the price is below the lower band, buy
    if the price is above the upper band, sell
4. Backtest trades based on the signal
'''
import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from analysis import cagr, sharpe_ratio


def download_data(ticker, start, end, overwrite=False):
    """
    Download or load data from Yahoo Finance for a given ticker and date range.

    Inputs:
    ticker: str, ticker symbol
    start: str, start date
    end: str, end date
    overwrite: bool, if True, download the data even if a local copy exists

    Returns:
    data: pd.DataFrame, data for the given ticker and date range
    """
    file_path = f'data/{ticker}_{start}_{end}.csv'

    # Make local data folder if it doesn't exist
    if not os.path.exists('data'):
        os.mkdir('data')

    # Check if the data is already downloaded and handle overwrite logic
    if os.path.exists(file_path) and not overwrite:
        print('Data already exists')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        print('Downloading data')
        data = yf.download(ticker, start, end, interval='1d')
        data.to_csv(file_path)
        data.index = pd.to_datetime(data.index)

    return data


def add_live_price(data, ticker):
    """
    Add the live price to the DataFrame if it's not already included.

    Inputs:
    data: pd.DataFrame, data
    ticker: str, ticker symbol

    Returns:
    data: pd.DataFrame, data with the live price added if it was missing
    """
    today_date = pd.Timestamp(datetime.today().date())

    if today_date not in data.index:
        print('Adding live price')
        ticker_yf = yf.Ticker(ticker)
        todays_data = ticker_yf.history(period='1d')

        # Check if today's data is not empty
        if not todays_data.empty:
            data.loc[today_date] = todays_data.iloc[0]
        else:
            print("No data available for today.")
    return data


def add_bollinger_bands(df, window=20, num_std=2):
    """
    Add Bollinger Bands to the DataFrame.

    Inputs:
    df: DataFrame object, data
    window: int, number of days to calculate the moving average
    num_std: int, number of standard deviations to calculate the bands
    """
    add_sma(df, window)
    sma_name = f'{window}d SMA'
    df['Upper Band'] = df[sma_name] + num_std * \
        df['Adj Close'].rolling(window=window).std()
    df['Lower Band'] = df[sma_name] - num_std * \
        df['Adj Close'].rolling(window=window).std()


def add_macd(df, short_window=12, long_window=26, signal_window=9):
    """
    Add MACD and Signal line to the DataFrame.

    Inputs:
    df: DataFrame object, data
    short_window: int, short period for the fast EMA
    long_window: int, long period for the slow EMA
    signal_window: int, period for the signal line EMA
    """
    df['EMA12'] = df['Adj Close'] \
        .ewm(span=short_window, min_periods=short_window).mean()
    df['EMA26'] = df['Adj Close'] \
        .ewm(span=long_window, min_periods=long_window).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal Line'] = df['MACD'] \
        .ewm(span=signal_window, min_periods=signal_window).mean()


def add_sma(df, window=20):
    """
    Add a simple moving average to the DataFrame.

    Inputs:
    df: DataFrame object, data
    window: int, number of days to calculate the moving average
    """
    sma_name = f'{window}d SMA'
    df[sma_name] = df['Adj Close'].rolling(window=window).mean()


def display_head(data):
    """
    Display the first 5 rows of the DataFrame.

    Inputs:
    data: pd.DataFrame, data
    """
    print(data.head())


def display_tail(data):
    """
    Display the last 5 rows of the DataFrame.

    Inputs:
    data: pd.DataFrame, data
    """
    print(data.tail())


def simulate_trades(df, signal_column):
    """
    Simulate trades based on the signal column.

    Inputs:
    df: DataFrame object, data
    signal_column: str, column name with the trading signal
    """
    # to excecute the trade the next day
    df['strategy'] = 0
    df[signal_column].shift(1)
    holding = False
    money = 1
    shares = 0
    for i in range(0, len(df)):
        if df[signal_column].iloc[i] == 1 and not holding:
            shares = money / df['Adj Close'].iloc[i]
            money = 0
            holding = True

        elif df[signal_column].iloc[i] == -1 and holding:
            money = shares * df['Adj Close'].iloc[i]
            shares = 0
            holding = False
        if holding:
            df['strategy'].iloc[i] = df['Adj Close'].iloc[i] * shares
        else:
            df['strategy'].iloc[i] = money
    df['strategy returns'] = df['strategy'].pct_change()


def add_signals(df, window=20, num_std=2):
    """
    Simple Bollinger Bands trading strategy for testing

    Inputs:
    df: DataFrame object, data

    """
    add_bollinger_bands(df, window, num_std)

    df['Signal'] = 0
    df.loc[df['Adj Close'] <= df['Lower Band'], 'Signal'] = 1
    df.loc[df['Adj Close'] >= df['Upper Band'], 'Signal'] = -1


if __name__ == '__main__':
    data = yf.download('SPY', '2023-08-14', '2024-08-14')
    add_signals(data, 20, 2)
    simulate_trades(data, 'Signal')
    print(data)

    print(f"CAGR: {cagr(data['strategy'])}")
    print(f"Sharpe: {sharpe_ratio(data['strategy returns'])}")
