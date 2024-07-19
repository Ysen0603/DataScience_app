import streamlit as st
from PG import data_loading, data_exploration, data_preprocessing, data_grouping, model_training, model_saving

def main():
    st.set_page_config(
        page_title="Analyse de données et ML",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Styles CSS personnalisés
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    h1 {
        color: #4CAF66;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .stSelectbox label {
        font-weight: bold;
        color: #31333F;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Analyse de données et apprentissage automatique")

    # Menu de navigation avec icônes
    menu = {
        "Chargement des données": "📥",
        "Exploration des données": "🔍",
        "Prétraitement": "🛠️",
        "Groupement": "🔗",
        "Entraînement du modèle": "🤖",
        "Sauvegarde du modèle": "💾"
    }
    choice = st.sidebar.selectbox(
        "Navigation",
        list(menu.keys()),
        format_func=lambda x: f"{menu[x]} {x}"
    )

    # Ajoutez une séparation visuelle
    st.sidebar.markdown("---")

    # Informations sur l'application
    st.sidebar.info("Cette application permet d'analyser des données et d'entraîner des modèles de machine learning.")

    # Barre de progression
    progress_mapping = {
        "Chargement des données": 1,
        "Exploration des données": 2,
        "Prétraitement": 3,
        "Groupement": 4,
        "Entraînement du modèle": 5,
        "Sauvegarde du modèle": 6
    }
    progress = st.progress(0)
    progress.progress(progress_mapping[choice] / len(menu))

    # Initialisation de l'état de session
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None

    # Navigation avec gestion d'état
    if choice == "Chargement des données":
        data_loading.load_data()
    elif choice == "Exploration des données":
        if st.session_state.data is not None:
            data_exploration.explore_data()
        else:
            st.warning("Veuillez d'abord charger des données.")
    elif choice == "Prétraitement":
        if st.session_state.data is not None:
            data_preprocessing.preprocess_data()
        else:
            st.warning("Veuillez d'abord charger des données.")
    elif choice == "Groupement":
        if st.session_state.data is not None:
            data_grouping.group_data()
        else:
            st.warning("Veuillez d'abord charger des données.")
    elif choice == "Entraînement du modèle":
        if st.session_state.data is not None:
            model_training.train_model_page()
        else:
            st.warning("Veuillez d'abord charger et prétraiter les données.")
    elif choice == "Sauvegarde du modèle":
        if st.session_state.model is not None:
            model_saving.save_load_model()
        else:
            st.warning("Veuillez d'abord entraîner un modèle.")

    # Pied de page
    st.markdown("---")
    st.markdown("Développé avec ❤️ par Ennaya Yassine")

if __name__ == "__main__":
    main()