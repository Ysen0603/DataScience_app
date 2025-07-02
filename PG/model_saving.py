import streamlit as st
import os
from utils.model_utils import save_model, load_model

def save_load_model():
    st.header("Model Saving and Loading")

    # Sauvegarde du modèle
    if 'model' in st.session_state and st.session_state.model is not None:
        st.subheader("Save Model")
        model_name = st.text_input("File name to save the model", "my_model.joblib")
        
        if st.button("Save Model"):
            save_path = os.path.join("models", "saved_models", model_name)
            save_model(
                model=st.session_state.model,
                filename=save_path,
                feature_columns=st.session_state.feature_columns,
                target_column=st.session_state.target_column,
                best_params=st.session_state.model.get_params()
            )
            st.success(f"Model saved successfully as: {model_name}")

    # Load Model
    st.subheader("Load a Model")
    saved_models_dir = os.path.join("models", "saved_models")
    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir)
    saved_models = os.listdir(saved_models_dir)
    if saved_models:
        selected_model = st.selectbox("Choose a model to load", saved_models)

        if st.button("Load Model"):
            load_path = os.path.join(saved_models_dir, selected_model)
            loaded_model_info = load_model(load_path)
            st.session_state.model = loaded_model_info['model']
            st.session_state.feature_columns = loaded_model_info['feature_columns']
            st.session_state.target_column = loaded_model_info['target_column']
            
            st.success(f"Model {selected_model} loaded successfully!")
            st.write("Information about the loaded model:")
            st.write(f"Features: {loaded_model_info['feature_columns']}")
            st.write(f"Target: {loaded_model_info['target_column']}")
            if loaded_model_info['best_params']:
                st.write("Best Parameters:")
                st.write(loaded_model_info['best_params'])
    else:
        st.info("No saved models found.")