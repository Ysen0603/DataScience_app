import streamlit as st
import pandas as pd
from utils.data_utils import load_excel_or_csv

def load_data():
    st.header("Chargement des données")
    
    if st.session_state.data is not None:
        df = st.session_state.data
    
    uploaded_file = st.file_uploader("Choisissez un fichier Excel ou CSV", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        try:
            df = load_excel_or_csv(uploaded_file)
            st.session_state.data = df
            st.success("Fichier chargé avec succès!")
            
            st.subheader("Aperçu des données")
            st.write(df.head())
            
            st.subheader("Informations sur le dataset")
            st.write(df.describe())
            
        except Exception as e:
            st.error(f"Une erreur s'est produite lors du chargement du fichier: {e}")
    elif st.session_state.data is not None:
        st.success("Données précédemment chargées disponibles.")
        st.subheader("Aperçu des données")
        st.write(st.session_state.data.head())
        
        st.subheader("Informations sur le dataset")
        try:
            st.write(st.session_state.data.describe())
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de l'affichage des informations sur le dataset: {e}")
        
    else:
        st.info("Veuillez télécharger un fichier pour commencer.")
        
    