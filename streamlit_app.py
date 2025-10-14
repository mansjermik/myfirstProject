import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# ========================
# Load Model and Defaults
# ========================

model = joblib.load('titanic_xgb_model')


# ======================
# Streamlit app code
# ======================

st.set_page_config(page_title="Titanic Dataset Prediction...!", layout="wide")
st.title("Titanic Dataset Prediction App")

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Data Visualization"])
