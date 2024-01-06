import investpy
import pandas as pd
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
import numpy as np

from tensorflow.keras.models import load_mode

# from keras.models import Sequential
# from keras.layers import Dense,Dropout,LSTM

start = '01/06/2009'
end = dt.datetime.now().strftime("%d/%m/%Y")

company = "PLC"
df = investpy.get_stock_historical_data(stock= company,country="VietNam", from_date=start,to_date = end)

df = pd.DataFrame(df)
print(df)