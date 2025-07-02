import streamlit as st
from utils.preprocessing_utils import encode_categorical, normalize_data, handle_missing_values

def preprocess_data():
    st.header("Data Preprocessing")

    if st.session_state.data is not None:
        df = st.session_state.data

        # Option to change the first row to column names
        change_first_row_to_header = st.checkbox("Use first row as column names")
        
        if change_first_row_to_header:
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            st.session_state.data = df
            st.success("First row used as column names successfully.")
            st.write("Preview of data with new column names:")
            st.write(df.head())

        st.subheader("Select Columns to Preprocess")
        columns_to_preprocess = st.multiselect("Choose columns to preprocess", df.columns.tolist())
        
        if columns_to_preprocess:
            preprocessing_options = st.multiselect("Choose preprocessing operations",
                                                   ["Categorical Variable Encoding",
                                                    "Data Normalization",
                                                    "Missing Value Handling"])
            
            if "Categorical Variable Encoding" in preprocessing_options:
                categorical_columns = [col for col in columns_to_preprocess if df[col].dtype == 'object']
                if categorical_columns:
                    encoding_method = st.selectbox("Choose encoding method", ["One-Hot Encoding", "Label Encoding"])
                else:
                    st.info("No categorical columns selected for encoding.")

            if "Data Normalization" in preprocessing_options:
                numeric_columns = [col for col in columns_to_preprocess if df[col].dtype in ['float64', 'int64']]
                if numeric_columns:
                    normalization_method = st.selectbox("Choose normalization method", ["StandardScaler", "MinMaxScaler"])
                else:
                    st.info("No numerical columns selected for normalization.")

            if "Missing Value Handling" in preprocessing_options:
                missing_columns = [col for col in columns_to_preprocess if df[col].isnull().any()]
                if missing_columns:
                    missing_method = st.selectbox("Choose missing value handling method",
                                                  ["Delete", "Mean", "Median", "Most Frequent Value"])
                else:
                    st.info("No missing values in selected columns.")

            if st.button("Apply Preprocessing"):
                try:
                    if "Categorical Variable Encoding" in preprocessing_options and categorical_columns:
                        df = encode_categorical(df, categorical_columns, encoding_method)
                        st.success("Encoding performed successfully!")

                    if "Data Normalization" in preprocessing_options and numeric_columns:
                        df = normalize_data(df, numeric_columns, normalization_method)
                        st.success("Normalization performed successfully!")

                    if "Missing Value Handling" in preprocessing_options and missing_columns:
                        df = handle_missing_values(df, missing_columns, missing_method)
                        st.success("Missing value handling performed successfully!")

                    st.session_state.data = df
                    st.write("Preview of preprocessed data:")
                    st.write(df.head())
                    st.write(df.isnull().sum())
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
        else:
            st.info("Please select at least one column to preprocess.")
    else:
        st.warning("No data has been loaded. Please load a file first in the 'Data Loading' section.")
    if st.button("Reset Data"):
            if 'original_data' in st.session_state:
                st.session_state.data = st.session_state.original_data.copy()
                st.success("Data reset to original state!")
                st.write("Preview of original data:")
                st.write(st.session_state.data.head())
            else:
                st.warning("No original data available for reset.")