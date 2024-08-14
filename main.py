from datetime import datetime
from dataframe import download_data, display_data, simulate_trades
from mean_reversion import bollinger_bands
import matplotlib.pyplot as plt
from analysis import cagr, sharpe_ratio

end = datetime.today().strftime('%Y-%m-%d')
data = download_data('AAPL', '2019-08-13', end)
bollinger_bands(data, 20, 2)
simulate_trades(data, 'Signal')
display_data(data)
print(data[data['Signal'] != 0])


# plt.plot(data['Adj Close'])
# plt.plot(data['Upper Band'], color='red')
# plt.plot(data['Lower Band'], color='green')
# plt.plot(data['20d SMA'], color='orange')
# plt.show()

print(f"CAGR: {cagr(data['strategy'])}")
