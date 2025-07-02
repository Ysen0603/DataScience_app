import streamlit as st
import pandas as pd

def group_data():
    st.header("Data Grouping")

    if 'data' in st.session_state and st.session_state.data is not None:
        df = st.session_state.data

        # Initialize session variables if they don't exist
        if 'group_columns' not in st.session_state:
            st.session_state.group_columns = []
        if 'agg_columns' not in st.session_state:
            st.session_state.agg_columns = []
        if 'agg_functions' not in st.session_state:
            st.session_state.agg_functions = {}

        st.subheader("Select Columns for Grouping")
        group_columns = st.multiselect("Choose columns for grouping", df.columns, st.session_state.group_columns)
        st.session_state.group_columns = group_columns

        if group_columns:
            st.subheader("Select Columns for Aggregation")
            agg_columns = st.multiselect("Choose columns for aggregation",
                                         [col for col in df.columns if col not in group_columns],
                                         st.session_state.agg_columns)
            st.session_state.agg_columns = agg_columns

            if agg_columns:
                st.subheader("Select Aggregation Functions")
                for col in agg_columns:
                    if col not in st.session_state.agg_functions:
                        st.session_state.agg_functions[col] = ["mean"]  # Default value
 
                    selected_funcs = st.multiselect(f"Choose aggregation functions for {col}",
                                                    ["mean", "sum", "count", "min", "max"],
                                                    st.session_state.agg_functions[col],
                                                    key=f"agg_func_{col}")
                    st.session_state.agg_functions[col] = selected_funcs

                if st.button("Perform Grouping"):
                    agg_dict = {col: st.session_state.agg_functions[col] for col in agg_columns}
                    grouped_df = df.groupby(group_columns).agg(agg_dict).reset_index()
                    st.session_state.grouped_data = grouped_df
                    st.success("Grouping performed successfully!")
                    st.write("Preview of grouped data:")
                    st.write(grouped_df.head())
            else:
                st.warning("Please select at least one column for aggregation.")

        if st.button("Reset Data"):
            if 'original_data' in st.session_state:
                st.session_state.data = st.session_state.original_data.copy()
                st.session_state.group_columns = []
                st.session_state.agg_columns = []
                st.session_state.agg_functions = {}
                if 'grouped_data' in st.session_state:
                    del st.session_state.grouped_data
                st.success("Data reset to original state!")
                st.write("Preview of original data:")
                st.write(st.session_state.data.head())
            else:
                st.warning("No original data available for reset.")
    else:
        st.warning("No data has been loaded. Please load a file first in the 'Data Loading' section.")

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame({'A': range(10), 'B': range(10, 20), 'C': range(20, 30)})  # Sample data for testing

if 'original_data' not in st.session_state:
    st.session_state.original_data = st.session_state.data.copy()

