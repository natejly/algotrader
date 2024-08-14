from dataframe import download_data, display_data, simulate_trades
from mean_reversion import bollinger_bands
from analysis import cagr, sharpe_ratio

data = download_data('SPY', '2023-08-14', '2024-08-14')
bollinger_bands(data, 20, 2)
simulate_trades(data, 'Signal')
display_data(data)


print(f"CAGR: {cagr(data['strategy'])}")
print(f"Sharpe: {sharpe_ratio(data['strategy returns'])}")
