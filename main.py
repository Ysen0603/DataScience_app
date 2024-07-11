import streamlit as st
from P import page_data_loading as data_loading
from P import page_data_exploration as data_exploration
from P import page_data_preprocessing as data_preprocessing
from P import page_data_grouping as data_grouping
from P import page_model_training as model_training
from P import page_model_saving as model_saving

def main():
    st.set_page_config(page_title="Analyse de données et ML", layout="wide")
    st.title("Application d'analyse de données et d'apprentissage automatique")

    # Menu de navigation
    menu = ["Chargement des données", "Exploration des données", "Prétraitement", "Groupement", "Entraînement du modèle", "Sauvegarde du modèle"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # Session state pour stocker les données
    if 'data' not in st.session_state:
        st.session_state.data = None

    # Navigation
    if choice == "Chargement des données":
        data_loading.load_data()
    elif choice == "Exploration des données":
        data_exploration.explore_data()
    elif choice == "Prétraitement":
        data_preprocessing.preprocess_data()
    elif choice == "Groupement":
        data_grouping.group_data()
    elif choice == "Entraînement du modèle":
        model_training.train_model_page()
    elif choice == "Sauvegarde du modèle":
        model_saving.save_load_model()

if __name__ == "__main__":
    main()
