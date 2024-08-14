from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from dataframe import add_bollinger_bands

def buy_and_hold(df):
    """
    Buy and hold strategy.

    Inputs:
    df: DataFrame object, data

    Returns:
    percent_return: float, percent return of the strategy
    """
    initial_price = df['Adj Close'].iloc[0]
    final_price = df['Close'].iloc[-1]
    percent_return = ((final_price - initial_price) / initial_price) * 100
    return percent_return



def bollinger_bands_strategy(df):
    """
    Bollinger Bands trading strategy.

    Inputs:
    df: DataFrame object, data

    Returns:
    percent_return: float, percent return of the strategy
    """
    add_bollinger_bands(df)

    df['Signal'] = 0
    df.loc[df['Adj Close'] <= df['Lower Band'], 'Signal'] = 1
    df.loc[df['Adj Close'] >= df['Upper Band'], 'Signal'] = -1

    return simulate_trades(df, 'Signal')



def bb_macd_strategy(df):
    """
    Combined Bollinger Bands and MACD trading strategy.

    Inputs:
    df: DataFrame object, data

    Returns:
    percent_return: float, percent return of the strategy
    """
    add_bollinger_bands(df)
    add_macd(df)

    df['Signal'] = 0
    df.loc[(df['Adj Close'] <= df['Lower Band']) & (df['MACD'] > df['Signal Line']), 'Signal'] = 1
    df.loc[(df['Adj Close'] >= df['Upper Band']) & (df['MACD'] < df['Signal Line']), 'Signal'] = -1

    return simulate_trades(df, 'Signal')


def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame.

    Inputs:
    df: DataFrame object, data

    Outputs:
    df: DataFrame object, data with new features added
    """
    if 'Adj Close' not in df.columns:
        print("DataFrame does not have the 'Adj Close' column. Using 'Close' instead.")
        df['Adj Close'] = df['Close']

    if len(df) < 50:
        print("Not enough data to calculate technical indicators.")
        return df

    df['SMA20'] = df['Adj Close'].rolling(window=20).mean()
    df['SMA50'] = df['Adj Close'].rolling(window=50).mean()
    df['Daily Return'] = df['Adj Close'].pct_change()
    df['Volatility'] = df['Adj Close'].rolling(window=20).std()

    df.dropna(inplace=True)
    return df


def ml_test(df):
    """
    Machine learning-based trading strategy using RandomForestClassifier.

    Inputs:
    df: DataFrame object, data

    Returns:
    percent_return: float, percent return of the strategy
    """
    df = add_technical_indicators(df)

    features = ['SMA20', 'SMA50', 'Daily Return', 'Volatility']
    X = df[features]
    y = (df['Adj Close'].shift(-1) > df['Adj Close']).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    df['ml_signal'] = 0
    df.loc[X_test.index, 'ml_signal'] = y_pred

    return simulate_trades(df, 'ml_signal')


def simulate_trades(df, signal_column):
    """
    Simulate trading based on a signal column.

    Inputs:
    df: DataFrame object, data
    signal_column: str, name of the column containing buy/sell signals

    Returns:
    percent_return: float, percent return of the strategy
    """
    shares = 0
    money = 100
    holding = False

    for i in range(1, len(df)):
        if df[signal_column].iloc[i] == 1 and not holding:
            holding = True
            shares = money / df['Adj Close'].iloc[i]
        elif df[signal_column].iloc[i] == 0 and holding:
            holding = False
            money = shares * df['Adj Close'].iloc[i]
            shares = 0

    if holding:
        money = shares * df['Close'].iloc[-1]

    percent_return = money - 100
    return percent_return
