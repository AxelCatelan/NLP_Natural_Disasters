import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#from NLP_Natural_Disasters.encoders import TextTokenization
#from NLP_Natural_Disasters.trainer import Trainer_Test

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Welcome to the NLP Natural Disasters project!"}

@app.get("/predict")
def predict(text):
    data = {
        'text': [text,],
    }
    X_pred = pd.DataFrame.from_dict(data)
    pipeline = joblib.load('model.joblib')
    y_pred = pipeline.predict(X_pred)
    return {'isdisaster': str(y_pred[0][0])}
