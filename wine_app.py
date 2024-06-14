
import streamlit as st

import pandas as pd
import numpy as np
from sklearn.svm import SVC

data = pd.read_csv("C:\Software\cleaned_wine3.csv")

st.title('Wine App')

volatile_acidity = st.slider('volatile acidity', min_value=float(data['volatile acidity'].min()),
                    max_value=float(data['volatile acidity'].max()),
                    value=float(data['volatile acidity'].mean()))

citric_acid = st.slider('citric acid', min_value=float(data['citric acid'].min()),
                    max_value=float(data['citric acid'].max()),
                    value=float(data['citric acid'].mean()))

density = st.slider('density', min_value=float(data['density'].min()),
                    max_value=float(data['density'].max()),step=0.001,
                    value=float(data['density'].mean()))


sulphates = st.slider('sulphates', min_value=float(data['sulphates'].min()),
                    max_value=float(data['sulphates'].max()),
                    value=float(data['sulphates'].mean()))


alcohol = st.slider('alcohol', min_value=float(data['alcohol'].min()),
                    max_value=float(data['alcohol'].max()),
                    value=float(data['alcohol'].mean()))

df = pd.DataFrame({
    'volatile acidity': [volatile_acidity],
    'citric acid': [citric_acid],
    'density': [density],
    'sulphates': [sulphates],
    'alcohol': [alcohol]
})

import pickle 

with open('C:\Software\Models\\wine3_model.pkl', 'rb') as f:
    model = pickle.load(f)

if st.button('Kalite Tahmini', key='predict'):
    pred = model.predict(df)
    st.write(pred)