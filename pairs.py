import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def analyze_pair(ticker1, ticker2, start, end, plot=False):
    # Download data
    data = pd.DataFrame(columns=[ticker1, ticker2])
    data[ticker1] = yf.download(ticker1, start, end)['Adj Close']
    data[ticker2] = yf.download(ticker2, start, end)['Adj Close']

    # Perform OLS regression
    model = sm.OLS(data[ticker1], sm.add_constant(data[ticker2])).fit()
    hedge_ratio = model.params[ticker2]
    print(f'Hedge Ratio = {hedge_ratio}')

    # Calculate spread
    spread = data[ticker1] - hedge_ratio * data[ticker2]

    # Plot spread if requested
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(spread, label='Spread')
        plt.axhline(spread.mean(), color='red', linestyle='--', label='Mean')
        plt.legend()
        plt.show()

    # Perform ADF test
    adf_result = adfuller(spread, maxlag=1)
    return adf_result


adf = analyze_pair('BLNK', 'NIO', '2023-08-14', '2024-08-15', plot=True)
if adf:
    print(f't-stat value = {adf[0]}')
    print('p-value =', adf[1])
    print('Critical values:')
    for key, value in adf[4].items():
        print(f'   {key}: {value}')
    if adf[0] < adf[4]['5%']:
        print('Reject the null hypothesis at the 5% level')
    if adf[0] < adf[4]['1%']:
        print('Reject the null hypothesis at the 1% level')
