import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split


def analyze_pair(ticker1, ticker2, start_date='2018-12-01', end_date='2022-03-31'):
    # Download the data for the two tickers
    data = pd.DataFrame(columns=[ticker1, ticker2])
    data[ticker1] = yf.download(ticker1, start_date, end_date)['Adj Close']
    data[ticker2] = yf.download(ticker2, start_date, end_date)['Adj Close']

    # Visualize the data
    plt.figure(figsize=(15, 7))
    plt.plot(data[ticker1], lw=1.5, label=f'Close Price of {ticker1}', color='red')
    plt.plot(data[ticker2], lw=1.5, label=f'Close Price of {ticker2}', color='#6CA6CD')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.xlabel('Dates')
    plt.ylabel('Price ($)')
    plt.title(f'Close prices for {ticker1} and {ticker2}')
    plt.grid(True)
    plt.show()

    # Create a train dataframe of the two assets
    train_close, test_close = train_test_split(data[ticker1], test_size=0.5, shuffle=False)
    train = pd.DataFrame()
    train[ticker1] = data[ticker1]
    train[ticker2] = data[ticker2]

    # Run OLS regression
    model = sm.OLS(train[ticker1], train[ticker2]).fit()
    print(f'Hedge Ratio = {model.params.iloc[0]}')

    # Calculate the spread
    spread = train[ticker1] - model.params.iloc[0] * train[ticker2]

    # Plot the spread
    # plt.figure(figsize=(15, 7))
    # plt.plot(spread, label="Spread")
    # plt.title(f"Pair's Spread between {ticker1} and {ticker2}")
    # plt.ylabel("Spread")
    # plt.grid(True)
    # plt.show()

    # Calculate and return the ADF test results
    return adfuller(spread, maxlag=1)


# Example usage
adf = analyze_pair('BLNK', 'NIO')
print(f't-stat value = {adf[0]}')
print('p-value =', adf[1])
print('Critical values:')
for key, value in adf[4].items():
    print(f'   {key}: {value}')
