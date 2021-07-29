import pandas as pd
import datetime
import spacy

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

def sentiment_extractor(start_date, end_date, keywords, path):
    temp = pd.read_csv(path + f'{keywords}_{start_date}_{end_date}.csv')

    temp['created_at'] = temp['created_at'].apply(split_date)
    temp['spacy_processing'] = temp['tweet'].apply(spacy_tokenizer)
    temp['spacy_processing'] = temp['spacy_processing'].apply(spacy_stopword)
    temp['spacy_processing'] = temp['spacy_processing'].apply(spacy_lemmatizer)
    temp['spacy_string'] = temp['spacy_processing'].apply(token_to_string)
    temp['textblob_score'] = temp['spacy_string'].apply(textblob_scoring)
    temp['NLTK_score'] = temp['spacy_string'].apply(nltk_scoring)

    g = temp.groupby('created_at')
    temp['like_weight'] = temp.likes_count / g.likes_count.transform("sum")
    temp['textblob_weight'] = temp['textblob_score'] * temp['like_weight']
    temp['NLTK_weight'] = temp['NLTK_score'] * temp['like_weight']

    TB_avg = pd.DataFrame(g.textblob_score.mean())
    NLTK_avg = pd.DataFrame(g.NLTK_score.mean())
    TB_weighted = pd.DataFrame(g.textblob_weight.sum())
    NLTK_weighted = pd.DataFrame(g.NLTK_weight.sum())
    final_df = pd.concat([TB_avg,NLTK_avg,TB_weighted,NLTK_weighted], axis=1)
    return final_df

folderpath = 'C:/Users/deku2/PycharmProjects/AppliedProject/Data/Tweets_backup/'

# Set time period and keywords
START_DATE = datetime.datetime(2018, 1, 1)
END_DATE = datetime.datetime(2018, 3, 31)
CSV_START = 'bitcoin'

dtrange = pd.date_range(start=START_DATE, end=END_DATE, freq='d')
months = pd.Series(dtrange.month)
starts, ends = months.ne(months.shift(1)), months.ne(months.shift(-1))
df = pd.DataFrame({'m_start': dtrange[starts].strftime('%Y-%m-%d'), 'm_end': dtrange[ends].strftime('%Y-%m-%d')})

for idx in range(len(df)):
    start_date = df['m_start'][idx]
    end_date = df['m_end'][idx]
    if idx == 0:
        final_df = sentiment_extractor(start_date, end_date, CSV_START, folderpath)
    else:
        final_df = final_df.append(sentiment_extractor(start_date, end_date, CSV_START, folderpath))
print(final_df)