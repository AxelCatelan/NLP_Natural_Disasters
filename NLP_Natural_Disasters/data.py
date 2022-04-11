import os
import re
import string
import unidecode
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


def remove_repeated_char(words):
    for w in words:
        if re.search(r'(.)\1{2}', w):
            words.remove(w)
    return words


def remove_rare_words(df, feature, rarity=1):
    words_count = pd.Series(' '.join(df[feature]).split()).value_counts() # Get all words in the whole df and the number of time they each appear
    rare_words = words_count[words_count.values <= rarity] # Words that are not common enough to be taken into account
    df[feature] = df[feature].apply(lambda text: ' '.join(w for w in text.split() if w not in rare_words)) # Reconstruct sentence without rare words
    return df

def clean_data(df, drop_keyword=True):
    '''
    Returns a cleaned data
    Default behavior is to drop "keyword" columns
    '''

    clean_df = df.drop(columns=['location'])
    if drop_keyword == True:
        clean_df = clean_df.drop(columns=['keyword'])

    clean_df['text'] = clean_df['text'].apply(lambda text: re.sub(r'http\S+', '', text)) # Remove urls in tweet
    clean_df['text'] = clean_df['text'].apply(lambda text: re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', '', text)) # Remove twitter handle
    clean_df['text'] = clean_df['text'].apply(remove_punctuation) # Remove all punctuation
    clean_df['text'] = clean_df['text'].apply(lambda text: ''.join(c for c in text if not c.isdigit())) # remove digits
    clean_df['text'] = clean_df['text'].apply(lambda text: unidecode.unidecode(text)) # Transform non ascii characters (accents, etc.) into their most corresponding character.
    clean_df['text'] = clean_df['text'].apply(lambda text: text.lower()) # Remove uppercase
    clean_df['text'] = clean_df['text'].apply(lambda text: text.strip()) # Remove useless spaces

    clean_df = remove_rare_words(clean_df, 'text') # Remove uncommon words

    clean_df['text'] = clean_df['text'].apply(remove_stopwords) # Remove stopwords and return a list of words
    clean_df['text'] = clean_df['text'].apply(remove_repeated_char) # Remove words with 3 or more repetition of the same character
    clean_df['text'] = clean_df['text'].apply(lemmatize_text) # Return a lemmatized text (change words to their roots)

    clean_df = clean_df[clean_df['text'] != ''] # Remove rows that ended up empty

    return clean_df


def get_data(addon=False, test=False):
    '''
    Returns the training data from the Kaggle Challenge
    with an optional parameter to also use additional data from another source
    '''
    if test==True:
        df = pd.read_csv(os.path.abspath(TEST_DATA_PATH))
        return df
    df = pd.read_csv(os.path.abspath(TRAIN_DATA_PATH))
    if addon == True:
        df_addon = pd.read_csv(ADDON_DATA_PATH)
        df = pd.concat([df, df_addon])
    return df


if __name__ == "__main__":
    df = get_data()
    print(df)
