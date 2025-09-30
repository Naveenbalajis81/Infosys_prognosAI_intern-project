import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Titanic Data Analysis")

@st.cache_data
def load_data():
    url="titanic.csv"
    data = pd.read_csv(url)
    return data

