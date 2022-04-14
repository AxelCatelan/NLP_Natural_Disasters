import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers, Sequential

from NLP_Natural_Disasters.data import get_data, clean_data
from NLP_Natural_Disasters.encoders import TextTokenization

class Trainer_Test(object):
    def __init__(self, X, y):
        self.pipeline = None
        self.X = X
        self.y = y


    def set_pipeline(self):
        c_df = clean_data(get_data())

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(c_df['text'])
        vocab_size = len(tokenizer.word_index)

        model = Sequential([
            layers.Embedding(input_dim=vocab_size + 1, output_dim=50 , mask_zero=True),
            layers.LSTM(20),
            layers.Dense(10, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['acc']
        )

        self.pipeline = Pipeline([
            ('preproc', TextTokenization(c_df)),
            ('model', model)
        ])


    def save_model_locally(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print("model.joblib saved locally")


    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)


if __name__ == "__main__":
    df = clean_data(get_data())
    X = df[['text']]
    y = df['target']
    trainer = Trainer_Test(X=X, y=y)
    trainer.set_pipeline()
    trainer.save_model_locally()
