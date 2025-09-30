import streamlit as st
import joblib
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from Trained_model import model_training

# Load the trained model
file="iris_model.pkl"
if file is None:
    model_training()
    st.info("Model trained and saved as iris_model.pkl")
    file="iris_model.pkl"
    
model = joblib.load(file)
iris = load_iris()
df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['species'] =[iris.target_names[i] for i in iris.target]

st.set_page_config(page_title="Iris Species Predictor",page_icon="🎴", layout="wide")

st.title("Iris Flower Species Prediction & Data Exploration App")
st.write("This app predicts the species of an Iris flower based on input measurements.")

# Sidebar for navigation
st.sidebar.title("Navigation")
mode=st.sidebar.radio("Choose Mode",["Prediction","Data Exploration"])

#mode for data exploration
if mode=="Data Exploration":
    st.subheader("Iris Dataset Overview")
    st.write("This section lets you explore the dataset visually and statistically.")
    
    if st.checkbox("Show Raw Data"):
        st.dataframe(df.head(10))
    
    feature_hist = st.selectbox("Select Feature for Histogram", iris.feature_names)
    fig,ax = plt.subplots()
    sns.histplot(df[feature_hist], kde=True, bins=20, ax=ax)
    st.pyplot(fig)
    
    st.write("### Scatter Plot")
    x_feature = st.selectbox("X-axis Feature", iris.feature_names, index=0)
    y_feature = st.selectbox("Y-axis Feature", iris.feature_names, index=1)
    
    fig2,ax2 = plt.subplots()
    sns.scatterplot(data=df, x=x_feature, y=y_feature, hue='species', palette="Set2", ax=ax2)
    st.pyplot(fig2)
    st.snow()
    
#mode for prediction
elif mode=="Prediction":
    st.subheader("Predict Iris Species")
    st.write("Input the measurements of the Iris flower to predict its species.")
    st.sidebar.header("Enter Flower Measurements")
    
    # Input
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

    # Prediction
    if st.sidebar.button("Predict"):
        with st.spinner("Predicting..."):
            time.sleep(1)  # Simulate a delay for better UX
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)
        species = ['Setosa', 'Versicolor', 'Virginica']
        st.success(f"The predicted species is: **{species[prediction[0]]}**")
        st.balloons()
        
# About section
st.markdown("""
---
### About
- This app is built using Streamlit
- model: RandomForestClassifier trained on the Iris dataset.
- dataset: Iris (Scikit-learn)
""")