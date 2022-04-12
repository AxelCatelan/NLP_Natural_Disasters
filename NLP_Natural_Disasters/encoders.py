import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from NLP_Natural_Disasters.data import clean_data

class TextTokenization(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, clean_df, max_length=25):
        self.clean_df = clean_df
        self.max_length = max_length
        self.vocab_size = -1

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.clean_df['text'])
        self.vocab_size = len(tokenizer.word_index)

        X_ = X.copy()
        X_clean = clean_data(X_, drop_keyword=False, drop_location=False, drop_rare=False)
        X_token = tokenizer.texts_to_sequences(X_clean['text'])
        X_pad = pad_sequences(X_token, maxlen=self.max_length, dtype='float32', padding='post')

        return X_pad
