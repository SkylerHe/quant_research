import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from scipy.fft import ifft

#load the file
dfsh = pd.read_csv('/Users/skylerhe/Desktop/SWS/sh000001.csv')
#check the file info.
#dfsh.info()
#categorize the days into months
dfsh['date_str']=dfsh['date'].astype(str)
#only collect the year and month of date
dfsh['dateym'] = dfsh['date_str'].str[:6]
#remove uncombine and unnecessary column(int64)
dfnew = dfsh.drop(['code','date','date_str','Unnamed: 0'],axis=1)
#split-apply-combine operation and get dfmean
#dfmean only includes dateym and close mean
dfmean = dfnew.groupby('dateym',as_index=False).mean()


# 1. Calculate the year-on-year percentage change
dfmean['YoY'] = dfmean['close'].pct_change(periods=12)
# Plot the year-on-year percentage change
plt.plot(dfmean['dateym'], dfmean['YoY'])
plt.title('Year-on-Year Percentage Change')
plt.xlabel('Date')
plt.ylabel('YoY Change')
# Show the plot 同比序列
plt.show()


#2.进行傅里叶变换，形成频谱图
yoy_data = dfmean['YoY'].dropna().tolist()
spectrum = np.abs(fft(yoy_data))
# 计算频率轴
frequency = np.fft.fftfreq(len(yoy_data))
# 绘制频谱图
plt.plot(frequency, spectrum)
plt.title('Spectrum of Year-on-Year Data')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


#3.find the top 1，5，10 maxima of frequency and apply Gaussian filter to smooth the YoY data
# a.Subtract the mean from the data to remove the DC component
yoy_data_no_dc = yoy_data - np.mean(yoy_data)
# Calculate the spectrum of the data without the DC component
spectrum_no_dc = np.abs(fft(yoy_data_no_dc))
# Find the indices of the top 10 maxima
top_maxima_indices = np.argsort(spectrum_no_dc)[-11:]

# b.Create complex signals for each of the top 1 and top 5 and top 10 frequencies
spectrum_filtered1 = np.zeros_like(spectrum)
spectrum_filtered1[top_maxima_indices[0] + 1] = spectrum[top_maxima_indices[0] + 1]
spectrum_filtered1[-top_maxima_indices[0] - 1] = spectrum[-top_maxima_indices[0] - 1]

spectrum_filtered2 = np.zeros_like(spectrum)
spectrum_filtered2[top_maxima_indices[4] + 1] = spectrum[top_maxima_indices[4] + 1]
spectrum_filtered2[-top_maxima_indices[4] - 1] = spectrum[-top_maxima_indices[4] - 1]

spectrum_filtered3 = np.zeros_like(spectrum)
spectrum_filtered3[top_maxima_indices[9] + 1] = spectrum[top_maxima_indices[9] + 1]
spectrum_filtered3[-top_maxima_indices[9] - 1] = spectrum[-top_maxima_indices[9] - 1]

# c.Transform back to the time domain using the inverse Fourier transform
filtered_signal1 = ifft(spectrum_filtered1)
filtered_signal2 = ifft(spectrum_filtered2)
filtered_signal3 = ifft(spectrum_filtered3)
# The results are complex numbers, take only the real parts
filtered_signal_real1 = np.real(filtered_signal1)
filtered_signal_real2 = np.real(filtered_signal2)
filtered_signal_real3 = np.real(filtered_signal3)
# Plot the filtered signals
plt.figure(figsize=(14, 10))
plt.plot(dfmean['dateym'][12:], filtered_signal_real1, label='Frequency 1')
plt.plot(dfmean['dateym'][12:], filtered_signal_real2, label='Frequency 5')
plt.plot(dfmean['dateym'][12:], filtered_signal_real3, label='Frequency 10')
plt.title('Filtered Signals at Top 1  5  10 Frequencies')
plt.xlabel('Date')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()



# 4. Fuse the three frequency signals into one
fused_signal = filtered_signal_real1 + filtered_signal_real2 + filtered_signal_real3
# Create a new figure
plt.figure(figsize=(14, 10))
# Plot the Year-on-Year percentage change
plt.plot(dfmean['dateym'], dfmean['YoY'], label='Year-on-Year Percentage Change')
# Plot the fused signal
plt.plot(dfmean['dateym'][12:], fused_signal, label='Fused Frequency 1, 5, 10')
# Add labels and title
plt.title('Year-on-Year Percentage Change and Fused Signal from Top 1, 5, 10 Frequencies')
plt.xlabel('Date')
plt.ylabel('Amplitude')
# Add a legend
plt.legend()
# Show the plot
plt.show()


