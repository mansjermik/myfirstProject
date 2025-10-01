import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("📊 Train a ML Model on Your CSV")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load CSV
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Select target column
    target_col = st.selectbox("Select the target column (label)", df.columns)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Only keep numeric features
        X = X.select_dtypes(include=["int64", "float64"])

        # Train-test split
        test_size = st.slider("Test size (as %)", 10, 50, 20) / 100.0
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Train model
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained! Accuracy: {acc:.2f}")

        # Feature importance
        st.subheader("Feature Importances")
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": clf.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importance)
