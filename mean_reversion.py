from dataframe import add_bollinger_bands


def bollinger_bands(df, window=20, num_std=2):
    """
    Simple Bollinger Bands trading strategy for testing

    Inputs:
    df: DataFrame object, data

    Returns:
    percent_return: float, percent return of the strategy
    """
    add_bollinger_bands(df, window, num_std)

    df['Signal'] = 0
    df.loc[df['Adj Close'] <= df['Lower Band'], 'Signal'] = 1
    df.loc[df['Adj Close'] >= df['Upper Band'], 'Signal'] = -1
