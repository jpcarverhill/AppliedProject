import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

def reformat_date(date):
    return (datetime.datetime.strptime(date, '%d/%m/%Y'))
def add_lag(df, column, lags):
    lag_columns = [f"{column}_lag{i}" for i in range(1,lags+1)]
    lag_values = [df[column].shift(i) for i in range(1,lags+1)]
    lag_df = pd.concat(lag_values,axis=1, keys=lag_columns)
    return df.join(lag_df)

folderpath = './PycharmProjects/AppliedProject/Data/Price/'
temp = pd.read_csv(folderpath + 'BTC-USD.csv')
temp = temp.set_index('Date')
temp = temp.rename(columns={'Volume':'Trade_Volume'})

# Impute missing data with Forward fill
temp = temp.fillna(method='ffill')
# Remove the extra price data
temp = temp[334:]
# Calculate daily return with percentage change of Close price
temp['Return'] = temp['Close'].pct_change()
# Standardize Close and Trade_Volume values between zero and one
temp[['Close','Trade_Volume']] = scaler.fit_transform(temp[['Close','Trade_Volume']])
# Compute volatility and moving average data
temp['3D_Volatility'] = temp['Return'].rolling(window=3).std()
temp['5D_Volatility'] = temp['Return'].rolling(window=5).std()
temp['7D_Volatility'] = temp['Return'].rolling(window=7).std()
temp['3D_MA'] = temp['Close'].rolling(window=3).mean()
temp['5D_MA'] = temp['Close'].rolling(window=5).mean()
temp['7D_MA'] = temp['Close'].rolling(window=7).mean()

# Introduce lagged data
lag = 3
temp = add_lag(temp, 'Return', lag)
temp = add_lag(temp, 'Trade_Volume', lag)
temp = add_lag(temp, '3D_Volatility', lag)
temp = add_lag(temp, '5D_Volatility', lag)
temp = add_lag(temp, '7D_Volatility', lag)
temp = add_lag(temp, '3D_MA', lag)
temp = add_lag(temp, '5D_MA', lag)
temp = add_lag(temp, '7D_MA', lag)

# Create two states, state zero when return is below 3%, state one otherwise
temp.loc[temp['Return'] <= 0.03, 'State'] = 0
temp.loc[temp['Return'] > 0.03, 'State'] = 1
temp['State_tmr'] = temp['State'].shift(-1)
# Remove the extra price data
temp = temp[31:1308]

final_df = temp.drop(columns=['Open','High','Low','Adj Close'])
final_df.to_csv(f'Processed_data/Processed_BTC-USD_withlags_3%.csv')