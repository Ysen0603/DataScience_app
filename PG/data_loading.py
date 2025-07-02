import streamlit as st
import pandas as pd
from utils.data_utils import load_excel_or_csv

def load_data():
    st.header("Data Loading")
    
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        try:
            df = load_excel_or_csv(uploaded_file)
            st.session_state.data = df
            st.session_state.original_data = df.copy()  # Stocke une copie des données originales
            st.success("File loaded successfully!")
            
            st.subheader("Data Preview")
            st.write(df.head())
            
            st.subheader("Dataset Information")
            st.write(df.describe())
            
        except Exception as e:
            st.error(f"An error occurred while loading the file: {e}")
    elif 'data' in st.session_state and st.session_state.data is not None:
        st.success("Previously loaded data available.")
        st.subheader("Data Preview")
        st.write(st.session_state.data.head())
        
        st.subheader("Dataset Information")
        try:
            st.write(st.session_state.data.describe())
        except Exception as e:
            st.error(f"An error occurred while displaying dataset information: {e}")
    else:
        st.info("Please upload a file to start.")

    