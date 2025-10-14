# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import xgboost
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score

# st.title("👽 Machine Learning Project App")
# st.info("Give the input to find out the result...")
# st.sidebar.title("Inputs")

import streamlit as st
import pandas as pd
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
