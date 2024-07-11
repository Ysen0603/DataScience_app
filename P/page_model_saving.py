import streamlit as st
import os
from utils.model_utils import save_model, load_model

def save_load_model():
    st.header("Sauvegarde et chargement de modèle")

    # Sauvegarde du modèle
    if 'model' in st.session_state and st.session_state.model is not None:
        st.subheader("Sauvegarde du modèle")
        model_name = st.text_input("Nom du fichier pour sauvegarder le modèle", "mon_modele.joblib")
        
        if st.button("Sauvegarder le modèle"):
            save_path = os.path.join("models", "saved_models", model_name)
            save_model(
                model=st.session_state.model,
                filename=save_path,
                feature_columns=st.session_state.feature_columns,
                target_column=st.session_state.target_column,
                best_params=st.session_state.model.get_params()
            )
            st.success(f"Modèle sauvegardé avec succès sous le nom : {model_name}")

    # Chargement du modèle
    st.subheader("Chargement d'un modèle")
    saved_models_dir = os.path.join("models", "saved_models")
    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir)
    saved_models = os.listdir(saved_models_dir)
    if saved_models:
        selected_model = st.selectbox("Choisissez un modèle à charger", saved_models)

        if st.button("Charger le modèle"):
            load_path = os.path.join(saved_models_dir, selected_model)
            loaded_model_info = load_model(load_path)
            st.session_state.model = loaded_model_info['model']
            st.session_state.feature_columns = loaded_model_info['feature_columns']
            st.session_state.target_column = loaded_model_info['target_column']
            
            st.success(f"Modèle {selected_model} chargé avec succès!")
            st.write("Informations sur le modèle chargé:")
            st.write(f"Caractéristiques : {loaded_model_info['feature_columns']}")
            st.write(f"Cible : {loaded_model_info['target_column']}")
            if loaded_model_info['best_params']:
                st.write("Meilleurs paramètres:")
                st.write(loaded_model_info['best_params'])
    else:
        st.info("Aucun modèle sauvegardé n'a été trouvé.")