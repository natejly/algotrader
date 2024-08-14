from datetime import datetime
from dataframe import download_data, display_data, simulate_trades
from mean_reversion import bollinger_bands
import matplotlib.pyplot as plt
from analysis import cagr, sharpe_ratio

end = datetime.today().strftime('%Y-%m-%d')
data = download_data('SPY', '2023-08-13', '2024-08-13')
bollinger_bands(data, 20, 2)
simulate_trades(data, 'Signal')
display_data(data)


print(f"CAGR: {cagr(data['strategy'])}")
print(f"Sharpe: {sharpe_ratio(data['Close'])}")

plt.plot(data['Adj Close'])
plt.plot(data['Upper Band'], color='red')
plt.plot(data['Lower Band'], color='green')
plt.plot(data['20d SMA'], color='orange')
plt.show()
