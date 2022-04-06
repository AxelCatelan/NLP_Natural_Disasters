import os
import re
import string
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

TRAIN_DATA_PATH = "../raw_data/train.csv"
ADDON_DATA_PATH = "../raw_data/tweets.csv"
TEST_DATA_PATH = "../raw_data/test.csv"


def remove_punctuation(text):
    for p in string.punctuation:
        text = text.replace(p, '')
    return text


def remove_stopwords(text, language='english'):
    stop_words = set(stopwords.words(language))
    return [w for w in word_tokenize(text) if not w in stop_words]


def lemmatize_text(text):
    lemmatizer  = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text])


def clean_data(df, drop_keyword=True):
    '''
    Returns a cleaned data
    Default behavior is to drop "keyword" columns
    '''

    clean_df = df.drop(columns=['location'])
    if drop_keyword == True:
        clean_df = clean_df.drop(columns=['keyword'])

    clean_df['text'] = clean_df['text'].apply(lambda text: re.sub(r'http\S+', '', text)) # Remove urls in tweet
    clean_df['text'] = clean_df['text'].apply(remove_punctuation) # Remove all punctuation / special characters
    clean_df['text'] = clean_df['text'].apply(lambda text: ''.join(c for c in text if not c.isdigit())) # remove digits
    clean_df['text'] = clean_df['text'].apply(lambda text: text.lower()) # Remove uppercase
    clean_df['text'] = clean_df['text'].apply(lambda text: text.strip()) # Remove useless spaces

    clean_df['text'] = clean_df['text'].apply(remove_stopwords) # Remove stopwords and return a list of words
    clean_df['text'] = clean_df['text'].apply(lemmatize_text) # Return a lemmatized text (change words to their roots)

    return clean_df


def get_data(addon=False):
    '''
    Returns the training data from the Kaggle Challenge
    with an optional parameter to also use additional data from another source
    '''
    df = pd.read_csv(os.path.abspath(TRAIN_DATA_PATH))
    if addon == True:
        df_addon = pd.read_csv(ADDON_DATA_PATH)
        df = pd.concat([df, df_addon])
    return df


if __name__ == "__main__":
    df = get_data()
    print(df.shape)
    df = get_data(addon=True)
    print(df.shape)
