from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Flatten, Embedding, Activation, Dropout, LSTM
#from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import numpy as np
from numpy import array
import pandas as pd

max_long = 25

def create_list(serie):
    text = serie.tolist()
    return text

def token_ize(text):
    token = Tokenizer()
    return token

def token_tweet(text, token):
    token.fit_on_texts(text)
    encoded_text = token.texts_to_sequences(text)
    X = pad_sequences(encoded_text, maxlen=max_long, padding='post')
    return X

def voc_token(token):
    vocab_size = len(token.word_index) + 1
    return vocab_size

def dict_token(token):
    dict_token_tweet = token.index_word.items()
    return dict_token_tweet

def glove_vector(token):
    glove_vectors = dict()
    #attention path à changer en fonction de l'emplacement du fichier.txt
    file = open('/content/gdrive/My Drive/glove.twitter.27B.200d.txt', encoding='utf-8')
    for line in file:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:])
        glove_vectors[word] = vectors
    file.close()
    word_vector_matrix = np.zeros((voc_token(token),200))
    to_delete = []
    for index, word in dict_token(token):
        vector = glove_vectors.get(word)
        if vector is not None:
            word_vector_matrix[index] = vector
        else :
            to_delete.append(word)
    return word_vector_matrix
