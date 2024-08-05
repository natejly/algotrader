# Description: Generates dataframe from yfinance and modifies it
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os


class dataframe:
    # Class to generate and modify dataframes
    def __init__(self, ticker, start, end, overwrite=False):
        """
        Get data from Yahoo Finance for a given ticker and date range
        Inputs
        ticker: str, ticker symbol
        start: str, start date
        end: str, end date
        Returns
        data: pd.DataFrame, data for the given ticker and date range
        """
        self.ticker = ticker
        self.start = start
        self.end = end
        self.window = 20
        self.file_path = f'data/{ticker}_{start}_{end}.csv'

        # make local data folder if it doesn't exist
        if not os.path.exists('data'):
            os.mkdir('data')

        # Check if the data is already downloaded and handle overwrite logic
        if os.path.exists(self.file_path) and not overwrite:
            print('Data already exists')
            self.data = pd.read_csv(self.file_path)
        else:
            print('Downloading data')
            self.data = yf.download(ticker, start=start, end=end)
            self.data.to_csv(self.file_path)

    def display(self):
        """
        Display the dataframe

        Inputs
        dataframe: pd.DataFrame, data
        """
        print(self.data)

    def head(self):
        """
        Get the first 5  rows of the dataframe

        Inputs
        dataframe: pd.DataFrame, data
        """
        print(self.data.head())

    def tail(self):
        """
        Get the last 5  rows of the dataframe

        Inputs
        dataframe: pd.DataFrame, data
        """
        print(self.data.tail())

    def plot(self, strategy=None):
        """
        Plot the data

        Inputs
        dataframe: pd.DataFrame, data
        """
        plt.title('Adjusted Close Price of ' + self.ticker)
        self.data['Adj Close'].plot()
        if strategy == 'bbounds':
            self.data['Upper Band'].plot(label='Upper Band', color='black')
            self.data['Lower Band'].plot(label='Lower Band', color='black')
            self.data[f'{self.window}d SMA'].plot(label=f'{self.window}d SMA',
                                                  color='orange')
            buy_signals = self.data[self.data[strategy] == 1].index
            sell_signals = self.data[self.data[strategy] == -1].index
            plt.plot(buy_signals, self.data.loc[buy_signals, 'Adj Close'],
                     '^', markersize=5, color='green', label='Buy Signal')
            plt.plot(sell_signals, self.data.loc[sell_signals, 'Adj Close'],
                     'v', markersize=5, color='red', label='Sell Signal')
        plt.show()
