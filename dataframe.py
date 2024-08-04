import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os
# Description: Generates dataframe from yfinance and modifies it


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

    def plot(self, bband=False, sma=False):
        """
        Plot the data

        Inputs
        dataframe: pd.DataFrame, data
        """
        plt.title('Adjusted Close Price of ' + self.ticker)
        self.data['Adj Close'].plot()
        if bband:
            self.data['Upper Band'].plot(label='Upper Band', color='red')
            self.data['Lower Band'].plot(label='Lower Band', color='green')
        if sma or bband:
            self.data[f'{self.window}d SMA'].plot(label=f'{self.window}d SMA',
                                                  color='orange')
        plt.legend()
        plt.show()

    def add_sma(self, window=20):
        """
        Add a simple moving average to the dataframe

        Inputs
        dataframe: pd.DataFrame, data
        window: int, number of days to calculate the moving average
        """
        name = f'{window}d SMA'
        self.data[name] = self.data['Adj Close'].rolling(window=window).mean()

    def add_bands(self, window=20, num_std=2):
        """
        Add Bollinger Bands to the dataframe

        Inputs
        dataframe: pd.DataFrame, data
        window: int, number of days to calculate the moving average
        num_std: int, number of standard deviations to calculate the bands
        """
        self.window = window
        sma_name = f'{window}d SMA'
        self.add_sma(window)
        self.data['Upper Band'] = self.data[sma_name] + \
            num_std * self.data['Adj Close'].rolling(window=window).std()
        self.data['Lower Band'] = self.data[sma_name] - \
            num_std * self.data['Adj Close'].rolling(window=window).std()


if __name__ == '__main__':
    apple = dataframe('AMD', '2023-01-01', '2024-08-04')
    apple.add_sma()
    apple.add_bands()
    apple.plot(bband=True)
    # apple.chart()
