import pandas as pd
import datetime
import spacy
import tqdm
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def split_date(date):
    return (datetime.datetime.strptime(date.split()[0], '%Y-%m-%d')) 

spacy_tokenizer = spacy.load('en_core_web_sm')

def spacy_stopword(text):
    return [token for token in text if not token.is_stop]

def spacy_lemmatizer(text):
    return [token.lemma_ for token in text]

def token_to_string(token):
    return ' '.join(token)

def textblob_scoring(text):
    return TextBlob(text).sentiment.polarity

sid = SentimentIntensityAnalyzer()
def nltk_scoring(text):
    return sid.polarity_scores(text)['compound']

def add_lag(df, column, lags):
    lag_columns = [f"{column}_lag{i}" for i in range(1,lags+1)]
    lag_values = [df[column].shift(i) for i in range(1,lags+1)]
    lag_df = pd.concat(lag_values,axis=1, keys=lag_columns)
    return df.join(lag_df)

def sentiment_extractor(start_date, end_date, keywords, path):
    temp = pd.read_csv(path + f'{keywords}_{start_date}_{end_date}.csv')

    temp['created_at'] = temp['created_at'].apply(split_date)
    temp.loc[temp['created_at']==end_date, 'created_at'] = datetime.datetime.strptime(end_date, '%Y-%m-%d') - datetime.timedelta(1)
    temp['spacy_processing'] = temp['tweet'].apply(spacy_tokenizer)
    temp['spacy_processing'] = temp['spacy_processing'].apply(spacy_stopword)
    temp['spacy_processing'] = temp['spacy_processing'].apply(spacy_lemmatizer)
    temp['spacy_string'] = temp['spacy_processing'].apply(token_to_string)
    temp['textblob_score'] = temp['spacy_string'].apply(textblob_scoring)
    temp['NLTK_score'] = temp['spacy_string'].apply(nltk_scoring)

    g = temp.groupby('created_at')
    temp['like_weight'] = temp.likes_count / g.likes_count.transform("sum")
    temp['textblob_weighted'] = temp['textblob_score'] * temp['like_weight']
    temp['NLTK_weighted'] = temp['NLTK_score'] * temp['like_weight']

    Replies = pd.DataFrame(g.replies_count.sum())
    Retweets = pd.DataFrame(g.retweets_count.sum())
    Likes = pd.DataFrame(g.likes_count.sum())
    Volume = pd.DataFrame(g.created_at.count())
    TB_avg = pd.DataFrame(g.textblob_score.mean())
    NLTK_avg = pd.DataFrame(g.NLTK_score.mean())
    TB_weighted = pd.DataFrame(g.textblob_weighted.sum())
    NLTK_weighted = pd.DataFrame(g.NLTK_weighted.sum())
    
    final_df = pd.concat([Replies,Retweets,Likes,Volume,TB_avg,NLTK_avg,TB_weighted,NLTK_weighted], axis=1)
    final_df = final_df.rename(columns={"replies_count": "Replies", "retweets_count": "Retweets","likes_count": "Likes", "created_at": "Volume","textblob_score": "textblob_average", "NLTK_score": "NLTK_average"})
    return final_df

# Set the folder path
folderpath = './PycharmProjects/AppliedProject/Data/Tweets/'

# Set time period and keywords
START_DATE = datetime.datetime(2018, 1, 1)
END_DATE = datetime.datetime(2021, 7, 1)
CSV_START = 'bitcoin'

# Create df of start and end date of every month between START_DATE and END_DATE
dtrange = pd.date_range(start=START_DATE, end=END_DATE, freq='d')
months = pd.Series(dtrange.month)
starts, ends = months.ne(months.shift(1)), months.ne(months.shift(-1))
df = pd.DataFrame({'m_start': dtrange[starts].strftime('%Y-%m-%d'), 'm_end': dtrange[ends].strftime('%Y-%m-%d')})

# Process Twitter data in monthly increments
for idx in tqdm.tqdm(range(len(df)-1)):
    start_date = df['m_start'][idx]
    end_date = df['m_start'][idx+1]
    if idx == 0:
        temp = sentiment_extractor(start_date, end_date, CSV_START, folderpath)
    else:
        temp = temp.append(sentiment_extractor(start_date, end_date, CSV_START, folderpath))

# Apply MinMaxScaler to dataset and scale data between zero and one
temp[:] = scaler.fit_transform(temp[:])

# Apply lag to data within dataset
lag = 3
for col in temp.columns:
    temp = add_lag(temp, col, lag)

# Save final dataframe to csv
final_df.to_csv(f'Processed_data/Processed_{CSV_START}_{START_DATE.strftime("%Y-%m-%d")}_{END_DATE.strftime("%Y-%m-%d")}.csv')