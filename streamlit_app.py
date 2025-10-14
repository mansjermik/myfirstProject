import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# ========================
# Load Model and Defaults
# ========================

model = joblib.load('titanic_xgb_model')


# ======================
# Streamlit app code
# ======================

st.set_page_config(page_title="Titanic Dataset Prediction...!", layout="wide")
st.title("Titanic Dataset Prediction App")
st.sidebar.title("User Inputs")
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Data Visualization"])

# ======== User Input =========
pclass = st.sidebar.slider('Select Passenger class', 1, 3, 2)
gender = st.sidebar.selectbox("Gender", ('male', 'female'))
sex = 1 if gender == 'male' else 0

age = st.sidebar.number_input("Age", min_value=1, max_value=100, step=1)

fare = st.sidebar.number_input("Fare", min_value=1, step=1)

sib = st.sidebar.select_slider("SibSp", (0, 1))

parch = st.sidebar.select_slider("Parch", (0, 1))

embarked = st.sidebar.selectbox("Embarked", ("S", "C", "Q"))

emb_s = emb_c = emb_q = 0

if embarked == 'S':
    emb_s = 1
elif embarked == 'Q':
    emb_q = 1
else:
    emb_c = 1


# =========== Creating dataframe of input values =============
data = {
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sib],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked_Q": [emb_q],
    "Embarked_S": [emb_s]
}

df = pd.DataFrame(data)


# ========== DATA VISUALIZATION ===========

with tab2:
    with st.expander("Show Selected Input Values"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f""" 
        Passenger class\n
        Gender\n
        Age\n
        Fare\n
        SibSp\n
        Parch\n
        Embarked\n
            """)

        with col_b:
            st.write(f""" 
        {pclass}\n
        {gender}\n
        {age}\n
        {fare}\n
        {sib}\n
        {parch}\n
        {embarked}\n
            """)


        if st.button("show data"):
            st.write(df.head(1))





# ======== Making Prediction ===========
with tab1:
    pred = model.predict(df).squeeze() 
    if pred == 1:
        st.success("Survived")
    else:
        st.warning("Didn't Survive")