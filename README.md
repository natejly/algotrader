# Algotrader

**Really complicated gambling**

## File Descriptions:

### `analysis.py`
- Runs backtesting analysis metrics for portfolio performance.

### `testing.py`
- Proof of concept file.
- Takes any ticker and date range and calculates 20-day SMA and Bollinger Bands.
- **Trading Strategy**:
  - **Sell**: If SMA is above the upper band.
  - **Buy**: If SMA is below the lower band.

### `pairs.py`
- **Pairs Trading**:
  1. **Generating Pairs**:
     - Takes in a list of stocks.
     - Checks for cointegration of them over a set window for initial filtering.
     - Note: This initial filter isn't strictly necessary since a sliding window is used to constantly check for cointegration, but it helps in maintaining system performance.

  2. **Generating Signals**:
     - Moves the sliding window into the "live" data.
     - Continuously checks if cointegration is maintained.
     - Dynamically updates z-scores and residuals.
     - Generate signals based on z score crossing thresholds
  3. **Backtesting**:
    - Does the calculations for returns and percent change
    - Gives metrics like total return, CAGR, Sharpe ratio





