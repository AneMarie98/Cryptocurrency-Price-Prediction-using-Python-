#Let's start with importing the libraries and the dataset. 
#For this task I will collect the latest Bitcoin prices data from Yahoo Finance, using the yfinance API. This will 
#help to collect the latest data each time we run this code:

import pandas as pd
import yfinance as yf
import datetime 
from datetime import date, timedelta
today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2= date.today() - timedelta(days=730)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

# Collecting the data

data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
print(data.head())

#In the code above we've collected the latest data of Bitcoin prices for the past
#730 days, and then we've prepared it for any data science task. 
#Let's have a look at the shape of this dataset to see if we're working with 730 rows or not. 

print(data.shape)

#Let's visualize the change in bitcoin prices till today by using a candlestick chart.

import plotly.graph_objects as go
figure = go.Figure(data=[go.Candlestick(x=data["Date"],open=data["Open"],high=data["High"],low=data["Low"],close=data["Close"])])
figure.update_layout(title = "Bitcoin Price Analysis", xaxis_rangeslider_visible=False)
figure.show()

#The Close column in the dataset contains the vlaues we need to predict. Let's have a look at the correlation of all the columns in the data concerning the Close column:

correlation = data.corr()
print(correlation["Close"].sort_values(ascending=False))

#CRYPTOCURRENCY PRICE PREDICTION MODEL 

#predicting the future prices of cryptocurrency is based on the problem of Time Series Analysis. The AutoTS library in Python is one of the best libraries
#for time series analysis. I will use this library to predict the bitcoin prices for the next 30 days. 

from autots import AutoTS
model = AutoTS(forecast_length=30, frequency='infer', ensemble='simple')
model = model.fit(data, date_col="Date", value_col="Close", id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)
