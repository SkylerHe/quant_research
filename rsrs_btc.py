# -*- coding: utf-8 -*-
"""RSRS_btc.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wBXmrUtqq9qN8OhclhR5N9Svyx_vj4v8

# Preprocessing the dataset
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
#load the datat from xlsx
dfbtc = pd.read_excel('/content/drive/My Drive/SWS_projects/btc.xlsx')
dfbtc

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

def calculate_slope(series):
    x = add_constant(np.arange(len(series)))
    model = OLS(series.values, x)
    result = model.fit()
    return result.params[1]

def rsrs_strategy(data, N1=18, N2=800, Z=1.0):
    data['H-L'] = data['high'] - data['low']
    data['beta'] = data['high'].rolling(N1).apply(calculate_slope)
    data['RSRS_R2'] = data['beta'].rolling(N2).mean()
    data['RSRS_Std'] = data['beta'].rolling(N2).std()
    data['Z-Score'] = (data['beta'] - data['RSRS_R2']) / data['RSRS_Std']

    # Generate signals
    data['signal'] = np.where(data['Z-Score'] > Z, 1, np.where(data['Z-Score'] < -Z, -1, 0))

    return data

#load the datat from xlsx
dfbtc = pd.read_excel('/content/drive/My Drive/SWS_projects/btc.xlsx')
rsrs_strategy(dfbtc)

import matplotlib.pyplot as plt

def plot_rsrs(data):
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

    # Plot the closing price and the signals
    axes[0].plot(data['close'], label='Close Price')
    axes[0].plot(data.loc[data['signal'] == 1].index, data['close'][data['signal'] == 1], '^', markersize=10, color='g', label='buy')
    axes[0].plot(data.loc[data['signal'] == -1].index, data['close'][data['signal'] == -1], 'v', markersize=10, color='r', label='sell')
    axes[0].set_ylabel('Price')
    axes[0].set_xlabel('Date')
    axes[0].set_title('RSRS Trading Signals')
    axes[0].legend()

    # Plot the RSRS Z-Score
    axes[1].plot(data['Z-Score'], label='RSRS Z-Score', color='b')
    axes[1].axhline(0, color='black', linestyle='--') # Add a horizontal line across the axis at 0.
    axes[1].axhline(1.0, color='red', linestyle='--') # Add a horizontal line across the axis at 1.0.
    axes[1].axhline(-1.0, color='green', linestyle='--') # Add a horizontal line across the axis at -1.0.
    axes[1].set_ylabel('Z-Score')
    axes[1].set_xlabel('Date')
    axes[1].set_title('RSRS Z-Score')
    axes[1].legend()

    plt.tight_layout() # Adjust the padding between and around the subplots.
    plt.show()

# Apply RSRS strategy
#df = rsrs_strategy(df)

# Plot RSRS
plot_rsrs(dfbtc)

import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have already loaded df_with_signals, which contains BTC data with RSRS and signals.
# If not, make sure to call the calculate_rsrs function first.

# Plotting BTC Price
plt.figure(figsize=(12, 6))
plt.plot(dfbtc['Date'], dfbtc['BTC_Price'], label='BTC Price', color='blue')

# Plotting RSRS indicator
plt.plot(dfbtc['Date'], dfbtc['RSRS'], label='RSRS', color='green', alpha=0.7)

# Plotting Buy signals
plt.scatter(dfbtc_with_signals[dfbtc_with_signals['Signal'] == 'Buy']['Date'],
            dfbtc_with_signals[dfbtc_with_signals['Signal'] == 'Buy']['BTC_Price'],
            marker='^', color='green', label='Buy Signal')

# Plotting Sell signals
plt.scatter(dfbtc_with_signals[dfbtc_with_signals['Signal'] == 'Sell']['Date'],
            dfbtc_with_signals[dfbtc_with_signals['Signal'] == 'Sell']['BTC_Price'],
            marker='v', color='red', label='Sell Signal')

plt.xlabel('Date')
plt.ylabel('BTC Price')
plt.title('BTC Price with RSRS Indicator and Buy/Sell Signals')
plt.legend()
plt.grid(True)
plt.show()
