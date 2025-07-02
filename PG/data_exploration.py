import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io


def explore_data():
    st.header("Data Exploration")

    if st.session_state.data is not None:
        df = st.session_state.data
        # Ajout des contrôles de taille dans la barre latérale
        st.sidebar.header("Chart Parameters")
        fig_width = st.sidebar.slider("Chart Width", 5, 20, 10)
        fig_height = st.sidebar.slider("Chart Height", 3, 10, 6)


        # Checkbox pour afficher les statistiques descriptives
        if st.checkbox("Show Descriptive Statistics"):
            st.subheader("Descriptive Statistics")
            st.write(df.describe())

        # Checkbox pour afficher les types de données
        if st.checkbox("Show Data Types"):
            st.subheader("Data Types")
            st.write(df.dtypes)

        # Checkbox pour afficher les valeurs manquantes
        if st.checkbox("Show Missing Values"):
            st.subheader("Missing Values")
            missing_data = df.isnull().sum()
            st.write(missing_data)

        # Corrélation entre les variables numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            if st.checkbox("Show Correlation Matrix"):
                st.subheader("Correlation between Numerical Variables")
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 3))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
        else:
            st.write("Not enough numerical columns to calculate correlation.")

        # Visualisation des données
        st.subheader("Data Visualization")
        graph_type = st.selectbox("Choose Chart Type",
                                  ["Histogram", "Bar Chart", "Box Plot", "Scatter Plot", "Pie Chart"])
        
        fig = None

        if graph_type == "Histogram":
            selected_columns = st.multiselect("Choose one or more columns", df.select_dtypes(include=[np.number]).columns)
            if selected_columns:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                for col in selected_columns:
                    sns.histplot(df[col], kde=True, ax=ax, label=col)
                plt.legend()
                plt.title("Histogram")
            else:
                st.warning("Please select at least one numerical column.")

        elif graph_type == "Bar Chart":
            selected_column = st.selectbox("Choose a column", df.columns)
            if selected_column:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                df[selected_column].value_counts().plot(kind='bar', ax=ax)
                plt.title("Bar Chart")
            else:
                st.warning("Please select a column.")

        elif graph_type == "Box Plot":
            selected_columns = st.multiselect("Choose one or more columns", df.select_dtypes(include=[np.number]).columns)
            if selected_columns:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                df[selected_columns].boxplot(ax=ax)
                plt.title("Box Plot")
            else:
                st.warning("Please select at least one numerical column.")

        elif graph_type == "Scatter Plot":
            x_col = st.selectbox("Choose column for X-axis", df.select_dtypes(include=[np.number]).columns)
            y_cols = st.multiselect("Choose one or more columns for Y-axis", df.select_dtypes(include=[np.number]).columns)
            if x_col and y_cols:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                for y_col in y_cols:
                    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, label=y_col)
                plt.legend()
                plt.title("Scatter Plot")
            else:
                st.warning("Please select a column for the X-axis and at least one column for the Y-axis.")

        elif graph_type == "Pie Chart":
            selected_column = st.selectbox("Choose a column", df.columns)
            if selected_column:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                data = df[selected_column].value_counts()
                wedges, _, autotexts = ax.pie(data.values,
                                            labels=data.index,
                                            autopct='%1.1f%%',
                                            textprops=dict(color="w"),
                                            startangle=90)
                ax.axis('equal')
                plt.setp(autotexts, size=8, weight="bold")
                plt.title(f"Distribution of {selected_column}")
                plt.legend(wedges, data.index,
                        title="Categories",
                        loc="center left",
                        bbox_to_anchor=(1, 0, 0.5, 1))
            else:
                st.warning("Please select a column.")

        # Display the chart
        if fig:
            st.pyplot(fig)
            
            # Option to export the chart
            if st.button("Export Chart"):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                
                st.download_button(
                    label="Download Chart",
                    data=buf,
                    file_name=f"{graph_type}.png",
                    mime="image/png"
                )

    else:
        st.warning("No data has been loaded. Please load a file first in the 'Data Loading' section.")