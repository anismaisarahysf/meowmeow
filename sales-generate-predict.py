import pickle
import streamlit as st
import pandas as pd
import seaborn as sns

st.write("#Simple Sales Prediction App")
st.write("This app predicts the **Sales** value!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0.00, 300.00, 75.00) #st.slider to generate slider
    Radio = st.sidebar.slider('Radio', 0.00, 9.90, 5.00)
    Newspaper = st.sidebar.slider('Newspaper', 0.00, 25.00, 5.00)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features() #this will rule out all the fx called abve

st.subheader('User Input parameters')
st.write(df)

loaded_model=pickle.load(open("Sales_Model.h5", "rb"))
pred = loaded_model.predict(df)

st.subheader('Sales Prediction')
st.write(pred)
