import pandas as pd
import streamlit as st
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import load_model
from NLP_Natural_Disasters.metrics import f1_m
from NLP_Natural_Disasters.data import clean_data

'''
# NLP Natural Disasters - Test
'''

data = st.text_area("Your tweet here!")

if data:
    mydict = {
        'text': [data,],
    }

    df = pd.DataFrame.from_dict(mydict)
    c_df = clean_data(df, drop_location=False, drop_keyword=False, drop_rare=False)
    print("Text :", c_df)
    text = c_df['text'].tolist()
    tokenizer = joblib.load('token_cam_test.joblib')
    encoded_text = tokenizer.texts_to_sequences(text)
    padded_text = pad_sequences(encoded_text, maxlen=25, padding='post')
    print(padded_text)
    model = load_model('RNN_keras_camille1', custom_objects={'f1_m': f1_m})
    y_pred = model.predict(padded_text)
    st.header("Prediction : " + str(y_pred[0][0]))

# TODO button load model / imports
# TODO Commenting!
# TODO classification Diana : show 3 classes with %
