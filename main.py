import streamlit as st
from PG import data_loading, data_exploration, data_preprocessing, data_grouping, model_training, model_saving

def main():
    st.set_page_config(
        page_title="Data Analysis and ML",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS styles
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
        color: #fd5c63;
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

    st.title("Data Analysis and Machine Learning")

    # Navigation menu with icons
    menu = {
        "Data Loading": "📥",
        "Data Exploration": "🔍",
        "Preprocessing": "🛠️",
        "Grouping": "🔗",
        "Model Training": "🤖",
        "Model Saving": "💾"
    }
    choice = st.sidebar.selectbox(
        "Navigation",
        list(menu.keys()),
        format_func=lambda x: f"{menu[x]} {x}"
    )

    # Add a visual separator
    st.sidebar.markdown("---")

    # Application information
    st.sidebar.info("This application allows you to analyze data and train machine learning models.")

    # Progress bar
    progress_mapping = {
        "Data Loading": 1,
        "Data Exploration": 2,
        "Preprocessing": 3,
        "Grouping": 4,
        "Model Training": 5,
        "Model Saving": 6
    }
    progress = st.progress(0)
    progress.progress(progress_mapping[choice] / len(menu))

    # Session state initialization
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None

    # Navigation with state management
    if choice == "Data Loading":
        data_loading.load_data()
    elif choice == "Data Exploration":
        if st.session_state.data is not None:
            data_exploration.explore_data()
        else:
            st.warning("Please load data first.")
    elif choice == "Preprocessing":
        if st.session_state.data is not None:
            data_preprocessing.preprocess_data()
        else:
            st.warning("Please load data first.")
    elif choice == "Grouping":
        if st.session_state.data is not None:
            data_grouping.group_data()
        else:
            st.warning("Please load data first.")
    elif choice == "Model Training":
        if st.session_state.data is not None:
            model_training.train_model_page()
        else:
            st.warning("Please load and preprocess data first.")
    elif choice == "Model Saving":
        if st.session_state.model is not None:
            model_saving.save_load_model()
        else:
            st.warning("Please train a model first.")

    # Footer
    st.markdown("---")
    st.markdown("Developed with ❤️ by Ennaya Yassine")

if __name__ == "__main__":
    main()