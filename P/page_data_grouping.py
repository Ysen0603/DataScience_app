import streamlit as st
import pandas as pd

def group_data():
    st.header("Groupement des données")

    if st.session_state.data is not None:
        df = st.session_state.data

        st.subheader("Sélection des colonnes pour le groupement")
        group_columns = st.multiselect("Choisissez les colonnes pour le groupement", df.columns)

        if group_columns:
            st.subheader("Sélection de la colonne pour l'agrégation")
            agg_column = st.selectbox("Choisissez la colonne pour l'agrégation", [col for col in df.columns if col not in group_columns])

            st.subheader("Sélection de la fonction d'agrégation")
            agg_function = st.selectbox("Choisissez la fonction d'agrégation", ["mean", "sum", "count", "min", "max"])

            if st.button("Effectuer le groupement"):
                grouped_df = df.groupby(group_columns)[agg_column].agg(agg_function).reset_index()
                st.session_state.data = grouped_df
                st.success("Groupement effectué avec succès!")
                st.write("Aperçu des données groupées:")
                st.write(grouped_df.head())
        else:
            st.info("Veuillez sélectionner au moins une colonne pour le groupement.")
    else:
        st.warning("Aucune donnée n'a été chargée. Veuillez d'abord charger un fichier dans la section 'Chargement des données'.")