# Runs dataframes through algorithms to generate trading signals

def add_sma(df, window=20):
    """
    Add a simple moving average to the dataframe

    Inputs
    dataframe: dataframe object, data
    window: int, number of days to calculate the moving average
    """
    name = f'{window}d SMA'
    df.data[name] = df.data['Adj Close'].rolling(window=window).mean()


def add_bands(df, window=20, num_std=2):
    """
    Add Bollinger Bands to the dataframe

    Inputs
    df: dataframe object, data
    window: int, number of days to calculate the moving average
    num_std: int, number of standard deviations to calculate the bands
    """
    df.window = window
    sma_name = f'{window}d SMA'
    add_sma(df, window)
    df.data['Upper Band'] = df.data[sma_name] + \
        num_std * df.data['Adj Close'].rolling(window=window).std()
    df.data['Lower Band'] = df.data[sma_name] - \
        num_std * df.data['Adj Close'].rolling(window=window).std()


def bbounds(df):
    """
    Buy when the price is below the lower band
    Sell when the price is above the upper band
    Inputs
    df: dataframe object, data
    Returns
    percent_return: float, percent return of the strategy
    """
    add_sma(df)
    add_bands(df)
    # Create new column for trading signals
    df.data['bbounds'] = 0

    # Buy when the price is below the lower band
    df.data.loc[df.data['Adj Close'] <= df.data['Lower Band'], 'bbounds'] = 1
    # Sell when the price is above the upper band
    df.data.loc[df.data['Adj Close'] >= df.data['Upper Band'], 'bbounds'] = -1
    # calculating returns with a state machine approach
    shares = 0
    money = 100
    holding = False

    for i in range(1, len(df.data)):
        if df.data['bbounds'].iloc[i] == 1 and not holding:
            holding = True
            shares = money / df.data['Adj Close'].iloc[i]
        elif df.data['bbounds'].iloc[i] == -1 and holding:
            holding = False
            money = shares * df.data['Adj Close'].iloc[i]
            shares = 0

    if holding:
        money = shares * df.data['Adj Close'].iloc[-1]
    percent_return = money - 100
    return percent_return
