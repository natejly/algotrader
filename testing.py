from dataframe import add_bollinger_bands, display_data, simulate_trades
from analysis import cagr, sharpe_ratio
import yfinance as yf


def bollinger_bands(df, window=20, num_std=2):
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
    bollinger_bands(data, 20, 2)
    simulate_trades(data, 'Signal')
    display_data(data)

    print(f"CAGR: {cagr(data['strategy'])}")
    print(f"Sharpe: {sharpe_ratio(data['strategy returns'])}")
