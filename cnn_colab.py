# -*- coding: utf-8 -*-
"""CNN_colab.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KyxDy85uq1Y6fUBmuXth0ZEBOCjqSRxV

# Preprocessing the dataset for CNN

## Load the dataset
"""

import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

# unzip the file and use '-o' to force overwrite the previous zipped file
!unzip -o '/content/drive/My Drive/SWS_projects/sh300_stock_price.npy.zip'

# Load the data from the .npy file
stockprice_data = np.load('sh300_stock_price.npy', allow_pickle=True)

#Basic info
#Each row in the represents a different stock at a certain date with some attributes.
#The columns appears to be
#Stock ID, Date, Opening_p, Highest_p, Lowest_p, Closing_p, volume

# Convert the numpy array to a pandas DataFrame
dfstockprice = pd.DataFrame(stockprice_data, columns=["Stock ID", "Date", "Open", "High", "Low", "Close", "Volume"])
dfstockprice.info()

"""## Preprocess the dataset"""

#Preprocessing the data before feeding it into a machine learning model
#typically converting everyting into numerical values
# Convert columns to their appropriate data types
dfstockprice['Date'] = pd.to_datetime(dfstockprice['Date'], format='%Y%m%d')
dfstockprice[['Open', 'High', 'Low', 'Close', 'Volume']] = dfstockprice[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric)
#To conduct a binary classification task using a CNN model on the stock price data
#To convert the stock price data into images (K-line charts) and then feeding these into CNN
#The label Y is the binary indicator of whether the cumulative return in the next 5D or 20D is positive (1) or non-positive (0)

#Feature Engineering - Moving Averages
#dfstockprice['Moving_Avg_20'] = dfstockprice.groupby('Stock ID')['Close'].transform(lambda x: x.rolling(window=20).mean())
#Feautre Engineering - Returns
#dfstockprice['Return_5D'] = dfstockprice.groupby('Stock ID')['Close'].transform(lambda x: x.shift(-5) / x - 1)
#dfstockprice['Return_20D'] = dfstockprice.groupby('Stock ID')['Close'].transform(lambda x: x.shift(-20) / x - 1)

# Create labels
#dfstockprice['Label_5D'] = np.where(dfstockprice['Return_5D'] > 0, 1, 0)
#dfstockprice['Label_20D'] = np.where(dfstockprice['Return_20D'] > 0, 1, 0)

#need datetime index
dfstockprice.set_index('Date')

"""# Image creation for CNN

## Image Creation

## Filter 2001 - 2019 training dataset
"""

#plot 2001 - 2019 dataset to be training set
# Filter data between 2001 and 2019 to be sample data
start_date = '2001-01-01'
end_date = '2019-12-31'
mask = (dfstockprice['Date'] >= start_date) & (dfstockprice['Date'] <= end_date)
df_sample_data = dfstockprice.loc[mask]



#set the index
df_sample_data = df_sample_data.set_index(pd.DatetimeIndex(df_sample_data['Date'].values))

dfstock = df_sample_data[df_sample_data['Stock ID'] == '000012.SZ'].copy()
dfstock


sampleID_list = df_sample_data['Stock ID'].unique().tolist()
sampleID_list[77:80]

"""## Plot every sequential 20-day candlestick charts for each stock"""

pip install pandas matplotlib mplfinance

import pandas as pd
import mplfinance as mpf
import os

def plot_candlestick (df, stockid, save_path):
  #check if the directory exists
  if not os.path.exists(save_path):
    #if it doesn't exist, create it
    os.makedirs(save_path)


  # Filter for the specific stock
  df_stock = df[df['Stock ID'] == stock_id].copy()

  # Calculate 20 day moving average
  df_stock['20_day_ma'] = df_stock['Close'].rolling(window=20).mean()

  # name for the image
  count = 0
  # Segement the original stock
  for i in range(0,len(df_stock),20):
    # Add MA to candlestick pattern
    moving_av = mpf.make_addplot(
                      df_stock.iloc[i:i+20]['20_day_ma'],
                      color = 'dodgerblue')


    #use mplfinance to build candlstick chart
    mpf.plot(
        df_stock.iloc[i:i+20],
        type = 'candle',
        style = 'yahoo',
        volume = True,
        axisoff = True,
        show_nontrading = False,
        addplot = moving_av,
        title = '',
        savefig = f'{save_path}/image_{count}.png'
    )
    count+=1

# Create stock id list
# Create stock id list
sampleID_list = df_sample_data['Stock ID'].unique().tolist()
for stock_id in sampleID_list[77:80]:
  save_path = f'/content/drive/My Drive/SWS_projects/Candlestick_charts/{stock_id}'
  plot_candlestick(df_sample_data,stock_id,save_path)



def plot_charts_for_stockID (df, stockID, save_dir):
    # Filter for one stock
    df_single_stock = df[df['Stock ID'] == stockID].copy()

    # Compute the moving average on the whole single stock data
    df_single_stock['Moving_Avg'] = df_single_stock['Close'].rolling(window=20).mean()

    # For each sequential 20-day window
    for i in range(0, len(df_single_stock) - 20 + 1, 20):  # increment by 20
        df_window = df_single_stock.iloc[i:i+20].copy()  # Create a copy to avoid SettingWithCopyWarning
        plot_candlestick_with_ma_mpf(df_window, stockID, i // 20, save_dir)  # save images as image_0, image_1, and so on

import time

start_time = time.time()  # Get the current time
# Specify the directory in Google Drive to save the plots
save_dir = "/content/drive/My Drive/SWS_projects/Candlestick_charts/"
for id in sampleID_list[:1]:
  plot_charts_for_stockID(df_sample_data, id, save_dir)

end_time = time.time()  # Get the current time again

elapsed_time = end_time - start_time  # Calculate the difference

elapsed_time/60

df_single_stock = df_sample_data[df_sample_data['Stock ID'] == '000001.SZ']

df_single_stock['Moving_Avg'] = df_single_stock.groupby('Stock ID')['Close'].transform(lambda x: x.rolling(window=5).mean())
df_single_stock.tail(40)

"""# Creating a Convolutional Neural Network (CNN)

### Call the ploting function
"""

# to figure out how long it will take to run
import time
start_time = time.time()

# plot all charts
for stock_id in sampleID_list[670:700]:
    save_path = '/content/drive/My Drive/SWS_projects/Candlestick_Charts/{}.png'.format(stock_id)
    plot_candlestick_with_ma(dfstockprice, 5, stock_id, save_path)

end_time = time.time()

elapsed_time = end_time - start_time
elapsed_time

"""## Transform images and split datasets"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from google.colab import drive
drive.mount('/content/drive')
import os

# Set the path to the directory containing the candlestick chart images
data_path = '/content/drive/My Drive/SWS_projects/Candlestick_Charts/'
print(os.getcwd())
# Define the transformation to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to a consistent size
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to the range [-1, 1]
])

# Load the dataset and apply the transformations
dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

# Split the dataset into training and testing sets
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])



"""## Build CNN Model

"""

# Define the CNN model architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 28 * 28, 2)  # Assuming binary classification (positive or non-positive return)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

"""## Train the model"""

# Create an instance of the CNN model
model = CNNModel()

# Set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set the batch size and number of workers for data loading
batch_size = 16
num_workers = 2

# Create data loaders for training and testing datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Set the number of epochs for training
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if (i+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Iteration {i+1}, Loss: {running_loss/10:.3f}")
            running_loss = 0.0

# Model evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")