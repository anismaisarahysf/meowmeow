import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

st.write("#Simple Sales Prediction App")
st.write("This app predicts the **Sales** value!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 4.3, 7.9, 5.4) #st.slider to generate slider
    Radio = st.sidebar.slider('Radio', 2.0, 4.4, 3.4)
    Newspaper = st.sidebar.slider('Newspaper', 1.0, 6.9, 1.3)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features() #this will rule out all the fx called abve

st.subheader('User Input parameters')
st.write(df)

data = sns.load_dataset('Sales')
X = data.drop(['Sales'],axis=1)
Y = data.Sales.copy()

modelGaussianSales = GaussianNB()
modelGaussianSales.fit(X, Y)

prediction = modelGaussianSales.predict(df)
prediction_proba = modelGaussianSales.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
