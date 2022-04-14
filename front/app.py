import pandas as pd
import streamlit as st
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

from tensorflow.keras.models import load_model
from NLP_Natural_Disasters.metrics import f1_m
from NLP_Natural_Disasters.data import clean_data

###
# FUNCTIONS
###

@st.cache(allow_output_mutation=True)
def get_tokenizer():
    """
    Return tokenizer, only executed once to load then returns cached tokenizer
    """
    tokenizer = joblib.load('token_cam_test.joblib')
    return tokenizer


@st.cache(allow_output_mutation=True)
def get_bow():
    """
    Return tokenizer, only executed once to load then returns cached tokenizer
    """
    bow = joblib.load('bow_dict.joblib')
    return bow


@st.cache(allow_output_mutation=True)
def get_RNN_model():
    """
    Return RNN Model, only executed once to load then returns cached model to avoid long prediction time
    """
    model = load_model('RNN_keras_camille1', custom_objects={'f1_m': f1_m})
    return model


@st.cache(allow_output_mutation=True)
def get_classification_model():
    """
    Return Classification Model, only executed once to load then returns cached model to avoid long prediction time
    """
    lda_model = LdaModel.load("model_lda")
    return lda_model


def preprocessing_tweet_rnn(tweet):
    """
    Preprocess a string for RNN : Clean the data, then tokenize and pad it
    """

    tweet_dict = {'text': [tweet,],} # Put the text in dict form to be compatible with our data cleaning format
    df = pd.DataFrame.from_dict(tweet_dict) # Tranform into pandas.DataFrame

    # Clean data
    # drop_location and drop_keyword = False to ignore those columns not present in this dataframe
    # drop_rare = False because we shouldn't do this on the text to predict
    c_df = clean_data(df, drop_location=False, drop_keyword=False, drop_rare=False)

    text = c_df['text'].tolist()
    encoded_text = tokenizer.texts_to_sequences(text) # Use the cached tokenizer to transform words into floats
    padded_text = pad_sequences(encoded_text, maxlen=25, padding='post') # Pad the encoded text to fit into the models
    return padded_text


def preprocessing_tweet_lda(tweet):
    """
    Preprocess a string for RNN : Clean the data, then tokenize and pad it
    """

    tweet_dict = {'text': [tweet,],} # Put the text in dict form to be compatible with our data cleaning format
    df = pd.DataFrame.from_dict(tweet_dict) # Tranform into pandas.DataFrame

    # Clean data
    # drop_location and drop_keyword = False to ignore those columns not present in this dataframe
    # drop_rare = False because we shouldn't do this on the text to predict
    c_df = clean_data(df, drop_location=False, drop_keyword=False, drop_rare=False)

    text = c_df['text'].tolist()
    sentence_spl = [x.split() for x in text]
    sentence_bowed = [bow.doc2bow(line) for line in sentence_spl]
    return sentence_bowed


def get_RNN_predict(tweet):
    """
    Takes a string and run it through preprocessing, then through the RNN model to return a prediction
    """
    padded_text = preprocessing_tweet_rnn(tweet)
    try:
        y_pred = RNNModel.predict(padded_text)
    except:
        return 0
    return round(y_pred[0][0])


def get_classification_predict(tweet):
    """
    Takes a string and run it through preprocessing, then through the classification model to return a % for each classes
    """
    proc_tweet = preprocessing_tweet_lda(tweet)
    predict_topic = LDAModel.get_document_topics(proc_tweet)
    percentage = [element for element in predict_topic]
    classes = [round(res[1] * 100, 2) for res in percentage[0]]
    return classes

# Loading the tokenizer and models a first time when running the streamlit
# to have the cached version available afterwards
tokenizer = get_tokenizer()
bow = get_bow()
RNNModel = get_RNN_model()
LDAModel = get_classification_model()


###
# FRONTEND
###

# Load and apply CSS style
with open('front/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

'''
# NLP Natural Disasters
'''

tweet = st.text_area(label="", max_chars=280)

if tweet != '':
    with st.spinner('Analysing...'):
        result = get_RNN_predict(tweet)
    if result == 0:
        st.success('''## All good👍''')
        st.balloons()
    else:
        st.error('''## This is a disaster''')
        get_classification_predict(tweet)
        class1, class2, class3 = get_classification_predict(tweet)
        col21, col22, col23 = st.columns(3)
        col21.write('## 🚒' + str(class1) + '%')
        col22.write('## 🌋' + str(class2) + '%')
        col23.write('## ❓' + str(class3) + '%')
else:
    st.info('Please enter your tweet above')
