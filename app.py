# app/app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure Streamlit page
st.set_page_config(page_title="User Risk Profiling Dashboard", layout="wide")
st.title("🔍 User Risk Profiling and Fraud Segmentation")

# Load Data
@st.cache_data
def load_data():
    file_path = "data/risk_scored_users.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

try:
    df = load_data()

    if df.empty:
        st.warning("🚨 The dataset is empty. Please check your CSV file.")
    else:
        # Sidebar filters
        st.sidebar.header("📊 Filter Options")
        risk_filter = st.sidebar.multiselect(
            "Select Risk Levels:", 
            options=df["risk_label"].unique(), 
            default=df["risk_label"].unique()
        )
        fraud_filter = st.sidebar.selectbox(
            "Select Fraud Type:", 
            options=["All", "Fraud", "Non-Fraud"]
        )

        # Apply filters
        filtered_df = df[df["risk_label"].isin(risk_filter)]
        if fraud_filter != "All":
            is_fraud = 1 if fraud_filter == "Fraud" else 0
            filtered_df = filtered_df[filtered_df["isFraud"] == is_fraud]

        # Data Preview
        st.subheader("🧾 Filtered Data Preview")
        st.dataframe(filtered_df.head(50))

        # Visualization 1: Risk Distribution
        st.subheader("📊 Risk Cluster Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=filtered_df, x="risk_label", palette="Set2", ax=ax1)
        st.pyplot(fig1)

        # Visualization 2: Fraud Count
        st.subheader("💰 Fraud Count by Risk Level")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=filtered_df, x="risk_label", hue="isFraud", palette="Set1", ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        st.pyplot(fig2)

        # Summary Statistics
        st.subheader("📈 Summary Statistics")
        st.write(filtered_df.describe())

except FileNotFoundError as fnf_error:
    st.error(f"❌ File Not Found: {fnf_error}")
except pd.errors.EmptyDataError:
    st.error("❌ The CSV file is empty.")
except Exception as e:
    st.error(f"❌ Failed to load or process data: {e}")
