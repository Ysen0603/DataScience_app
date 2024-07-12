import streamlit as st
import pandas as pd

def group_data():
    st.header("Groupement des données")

    if 'data' in st.session_state and st.session_state.data is not None:
        df = st.session_state.data

        # Initialiser les variables de session si elles n'existent pas
        if 'group_columns' not in st.session_state:
            st.session_state.group_columns = []
        if 'agg_columns' not in st.session_state:
            st.session_state.agg_columns = []
        if 'agg_functions' not in st.session_state:
            st.session_state.agg_functions = {}

        st.subheader("Sélection des colonnes pour le groupement")
        group_columns = st.multiselect("Choisissez les colonnes pour le groupement", df.columns, st.session_state.group_columns)
        st.session_state.group_columns = group_columns

        if group_columns:
            st.subheader("Sélection des colonnes pour l'agrégation")
            agg_columns = st.multiselect("Choisissez les colonnes pour l'agrégation", 
                                         [col for col in df.columns if col not in group_columns], 
                                         st.session_state.agg_columns)
            st.session_state.agg_columns = agg_columns

            if agg_columns:
                st.subheader("Sélection des fonctions d'agrégation")
                for col in agg_columns:
                    if col not in st.session_state.agg_functions:
                        st.session_state.agg_functions[col] = ["mean"]  # Default value

                    selected_funcs = st.multiselect(f"Choisissez les fonctions d'agrégation pour {col}", 
                                                    ["mean", "sum", "count", "min", "max"],
                                                    st.session_state.agg_functions[col],
                                                    key=f"agg_func_{col}")
                    st.session_state.agg_functions[col] = selected_funcs

                if st.button("Effectuer le groupement"):
                    agg_dict = {col: st.session_state.agg_functions[col] for col in agg_columns}
                    grouped_df = df.groupby(group_columns).agg(agg_dict).reset_index()
                    st.session_state.grouped_data = grouped_df
                    st.success("Groupement effectué avec succès!")
                    st.write("Aperçu des données groupées:")
                    st.write(grouped_df.head())
            else:
                st.warning("Veuillez sélectionner au moins une colonne pour l'agrégation.")

        if st.button("Réinitialiser les données"):
            if 'original_data' in st.session_state:
                st.session_state.data = st.session_state.original_data.copy()
                st.session_state.group_columns = []
                st.session_state.agg_columns = []
                st.session_state.agg_functions = {}
                if 'grouped_data' in st.session_state:
                    del st.session_state.grouped_data
                st.success("Données réinitialisées à l'état original!")
                st.write("Aperçu des données originales:")
                st.write(st.session_state.data.head())
            else:
                st.warning("Aucune donnée originale disponible pour la réinitialisation.")
    else:
        st.warning("Aucune donnée n'a été chargée. Veuillez d'abord charger un fichier dans la section 'Chargement des données'.")

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame({'A': range(10), 'B': range(10, 20), 'C': range(20, 30)})  # Sample data for testing

if 'original_data' not in st.session_state:
    st.session_state.original_data = st.session_state.data.copy()

