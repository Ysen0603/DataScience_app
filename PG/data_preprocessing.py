import streamlit as st
from utils.preprocessing_utils import encode_categorical, normalize_data, handle_missing_values

def preprocess_data():
    st.header("Prétraitement des données")

    if st.session_state.data is not None:
        df = st.session_state.data

        # Option to change the first row to column names
        change_first_row_to_header = st.checkbox("Utiliser la première ligne comme nom des colonnes")
        
        if change_first_row_to_header:
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            st.session_state.data = df
            st.success("La première ligne a été utilisée comme nom des colonnes.")
            st.write("Aperçu des données avec les nouveaux noms de colonnes:")
            st.write(df.head())

        st.subheader("Sélection des colonnes à prétraiter")
        columns_to_preprocess = st.multiselect("Choisissez les colonnes à prétraiter", df.columns.tolist())
        
        if columns_to_preprocess:
            preprocessing_options = st.multiselect("Choisissez les opérations de prétraitement", 
                                                   ["Encodage des variables catégorielles", 
                                                    "Normalisation des données", 
                                                    "Traitement des valeurs manquantes"])
            
            if "Encodage des variables catégorielles" in preprocessing_options:
                categorical_columns = [col for col in columns_to_preprocess if df[col].dtype == 'object']
                if categorical_columns:
                    encoding_method = st.selectbox("Choisissez la méthode d'encodage", ["One-Hot Encoding", "Label Encoding"])
                else:
                    st.info("Aucune colonne catégorielle sélectionnée pour l'encodage.")

            if "Normalisation des données" in preprocessing_options:
                numeric_columns = [col for col in columns_to_preprocess if df[col].dtype in ['float64', 'int64']]
                if numeric_columns:
                    normalization_method = st.selectbox("Choisissez la méthode de normalisation", ["StandardScaler", "MinMaxScaler"])
                else:
                    st.info("Aucune colonne numérique sélectionnée pour la normalisation.")

            if "Traitement des valeurs manquantes" in preprocessing_options:
                missing_columns = [col for col in columns_to_preprocess if df[col].isnull().any()]
                if missing_columns:
                    missing_method = st.selectbox("Choisissez la méthode de traitement des valeurs manquantes", 
                                                  ["Supprimer", "Moyenne", "Médiane", "Valeur la plus fréquente"])
                else:
                    st.info("Aucune valeur manquante dans les colonnes sélectionnées.")

            if st.button("Appliquer le prétraitement"):
                try:
                    if "Encodage des variables catégorielles" in preprocessing_options and categorical_columns:
                        df = encode_categorical(df, categorical_columns, encoding_method)
                        st.success("Encodage effectué avec succès!")

                    if "Normalisation des données" in preprocessing_options and numeric_columns:
                        df = normalize_data(df, numeric_columns, normalization_method)
                        st.success("Normalisation effectuée avec succès!")

                    if "Traitement des valeurs manquantes" in preprocessing_options and missing_columns:
                        df = handle_missing_values(df, missing_columns, missing_method)
                        st.success("Traitement des valeurs manquantes effectué avec succès!")

                    st.session_state.data = df
                    st.write("Aperçu des données prétraitées:")
                    st.write(df.head())
                except Exception as e:
                    st.error(f"Erreur lors du prétraitement: {e}")
        else:
            st.info("Veuillez sélectionner au moins une colonne à prétraiter.")
    else:
        st.warning("Aucune donnée n'a été chargée. Veuillez d'abord charger un fichier dans la section 'Chargement des données'.")
    if st.button("Réinitialiser les données"):
            if 'original_data' in st.session_state:
                st.session_state.data = st.session_state.original_data.copy()
                st.success("Données réinitialisées à l'état original!")
                st.write("Aperçu des données originales:")
                st.write(st.session_state.data.head())
            else:
                st.warning("Aucune donnée originale disponible pour la réinitialisation.")