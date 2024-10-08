# functions for analysis of the trading strategies
import numpy as np


def cagr(returns):
    """
    Function to calculate the Compound Annual Growth Rate (CAGR)

    Inputs:
    returns: pd.Series, daily returns

    Returns:
    float, CAGR
    """
    num_years = (returns.index[-1] - returns.index[0]).days / 365
    cagr_value = (returns.iloc[-1] / returns.iloc[0]) ** (1 / num_years) - 1
    return cagr_value * 100


def kelly(win_prob, win_size, loss_size):
    """
    Function to calculate the Kelly criterion

    Inputs:
    win_prob: float, probability of winning
    win_size: float, size of winning trades
    loss_size: float, size of losing trades

    Returns:
    float, Kelly criterion
    """
    return (win_prob - (1 - win_prob)) / (win_size / loss_size)


def sharpe_ratio(returns, risk_free_rate=0):
    """
    Function to calculate the Sharpe Ratio

    Inputs:
    returns: pd.Series, daily returns of the investment
    risk_free_rate: float, the risk-free rate (annualized)

    Returns:
    float, Sharpe Ratio
    """
    # doesn't work
    mean = returns.mean() * 252 - risk_free_rate
    sigma = returns.std(ddof=1) * np.sqrt(252)
    return mean / sigma


def calmar_ratio(returns):
    """
    Function to calculate the Calmar ratio

    Inputs:
    returns: pd.Series, daily returns

    Returns:
    float, Calmar ratio
    """
    max_drawdown = (1 - returns.div(returns.cummax())).max()
    return returns.mean() / max_drawdown
