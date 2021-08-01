import time
from datetime import datetime
import pandas as pd
import numpy as np
import twint

def scrape(start_date: str, end_date: str, keywords: str):
    config = twint.Config()
    config.Search = keywords
    config.Lang = "en"
    config.Min_likes = 3
    config.Since = start_date
    config.Until = end_date
    # config.Limit = 100 for testing

    config.Hide_output = True
    config.Store_csv = True
    config.Output = f"Tweets/{keywords}_{start_date}_{end_date}.csv"

    temp_time = time.time()
    twint.run.Search(config)
    runtime_seconds = time.time() - temp_time
    runtime_minutes = np.round(runtime_seconds / 60, 1)
    print(f"Downloaded the data in {runtime_minutes} minutes")
    return float(runtime_minutes)

# Set time period and keywords
START_DATE = datetime(2018, 1, 1)
END_DATE = datetime(2021, 7, 1)
KEYWORDS = 'bitcoin'

dtrange = pd.date_range(start=START_DATE, end=END_DATE, freq='d')
months = pd.Series(dtrange.month)
starts, ends = months.ne(months.shift(1)), months.ne(months.shift(-1))
df = pd.DataFrame({'m_start': dtrange[starts].strftime('%Y-%m-%d'), 'm_end': dtrange[ends].strftime('%Y-%m-%d')})

time_list = []
for idx in range(len(df)-1):
    start_date = df['m_start'][idx]
    end_date = df['m_start'][idx+1]
    time_list.append(scrape(start_date,end_date,KEYWORDS))
print(time_list)