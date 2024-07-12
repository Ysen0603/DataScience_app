import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data():
    st.header("Exploration des données")

    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.subheader("Statistiques descriptives")
        st.write(df.describe())

        st.subheader("Types de données")
        st.write(df.dtypes)

        st.subheader("Valeurs manquantes")
        missing_data = df.isnull().sum()
        st.write(missing_data)

        st.subheader("Corrélation entre les variables numériques")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.write("Pas assez de colonnes numériques pour calculer la corrélation.")

        st.subheader("Distribution des variables")
        selected_column = st.selectbox("Choisissez une colonne pour voir sa distribution", df.columns)
        
        if df[selected_column].dtype in ['int64', 'float64']:
            fig, ax = plt.subplots()
            sns.histplot(df[selected_column], kde=True, ax=ax)
            st.pyplot(fig)
        elif df[selected_column].dtype == 'object':
            fig, ax = plt.subplots()
            df[selected_column].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
        else:
            st.write("Type de données non pris en charge pour la visualisation.")

    else:
        st.warning("Aucune donnée n'a été chargée. Veuillez d'abord charger un fichier dans la section 'Chargement des données'.")