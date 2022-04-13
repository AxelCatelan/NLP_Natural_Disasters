import pandas as pd
import streamlit as st
import joblib

'''
# NLP Natural Disasters - Test
'''

text = st.text_area("Your tweet here!")

if text:
    data = {
        'text': [text,],
    }

    X_pred = pd.DataFrame.from_dict(data)
    pipeline = joblib.load('model.joblib')
    y_pred = pipeline.predict(X_pred)
    st.header("Prediction : " + str(y_pred[0][0]))
