from copyreg import pickle
import streamlit as st

st.title('Iris Species Predictor')
st.header("Let's predict Iris species")
st.subheader("cool app")

import plotly.express as px

@st.cache()
def read_data():
    return px.data.iris()

df_iris = read_data()

hist_sl = px.histogram(df_iris, x='sepal_length')

hist_sl

show_df = st.checkbox("do you want to see the data?")

if show_df:
    df_iris

sl = st.number_input("Sepal Length (cm)", 0, 100)
sw = st.number_input("Sepal Width (cm)", 0, 100)
pl = st.number_input("Petal Length (cm)", 0, 100)
pw = st.number_input("Petal Width (cm)", 0, 100)

import numpy as np

user_input = np.array([sl, sw, pl, pw])
user_input

import pickle

with open("saved-iris-model.pkl", "rb") as f:
    classifier = pickle.load(f)

import time 

from predict import predict_flower
with st.spinner('Predicting your iris species'):
    time.sleep(4)
    prediction = predict_flower(classifier, user_input)
    # prediction = classifier.predict([user_input])

st.header(f"The model predicts {prediction[0]}!")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("A Cat!")
    st.image('https://cdn.pixabay.com/photo/2017/11/09/21/41/cat-2934720_960_720.jpg', caption='Cute kitty!')
with col2:
    st.header("A Dog!")
    st.image('https://cdn.pixabay.com/photo/2019/08/19/07/45/dog-4415649_960_720.jpg', caption='Good boy!')

st.balloons()

st.sidebar.button("CLick here")

st.header("Hellloooo!!!!")