# main file to run strategies

from dataframe import dataframe
from algos import bbounds, hold


def backtest(ticker, start, end, strategy):
    """
    Backtest a strategy on a given ticker and date range

    Inputs
    ticker: str, ticker symbol
    start: str, start date
    end: str, end date
    """
    test = dataframe(ticker, start, end)
    print(f"{strategy.__name__} Percent Return: {round(strategy(test),4)}%")
    print(f"Market Percent Return: {round(hold(test),4)}%")
    test.plot(strategy=strategy.__name__)


if __name__ == '__main__':
    backtest('BLUE', '2023-01-01', '2024-04-01', bbounds)
