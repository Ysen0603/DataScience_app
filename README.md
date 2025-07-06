# Data Analysis and Machine Learning Streamlit Application

This interactive Streamlit application provides a comprehensive platform for data analysis and machine learning. It allows users to load, explore, preprocess, group data, train various machine learning models, and save/load them for future use.

## Features

*   **Data Loading**: Easily upload and preview your datasets in Excel (.xlsx) or CSV (.csv) formats.
*   **Data Exploration**: Gain insights into your data with descriptive statistics, data type displays, missing value summaries, correlation matrices, and a variety of interactive visualizations including histograms, bar charts, box plots, scatter plots, and pie charts.
*   **Data Preprocessing**: Prepare your data for modeling with options to:
    *   Use the first row as column headers.
    *   Handle missing values using methods like dropping rows/columns, mean, median, or most frequent imputation.
    *   Encode categorical variables using One-Hot Encoding or Label Encoding.
    *   Normalize numerical data using StandardScaler or MinMaxScaler.
*   **Data Grouping**: Aggregate your data by selected columns and apply various aggregation functions (mean, sum, count, min, max).
*   **Model Training**: Train machine learning models with a user-friendly interface:
    *   Select target and feature columns.
    *   Choose between classification (Logistic Regression, Decision Tree, Random Forest) and regression (Linear Regression, Decision Tree, Random Forest) tasks.
    *   Configure the test set size.
    *   Optionally perform hyperparameter optimization using GridSearch.
    *   Optionally perform cross-validation to assess model robustness.
*   **Model Evaluation & Visualization**: Evaluate trained models with key metrics and visualizations:
    *   Display accuracy (for classification) or Mean Squared Error (MSE) and R² score (for regression).
    *   Visualize classification results with ROC curves.
    *   Visualize regression results with residuals plots.
    *   Display feature importance for applicable models.
*   **Model Saving & Loading**: Save your trained models along with their feature and target column information, and load previously saved models for deployment or further analysis.

## Technologies Used

*   Python
*   Streamlit
*   Pandas
*   NumPy
*   Matplotlib
*   Seaborn
*   Scikit-learn
*   Joblib
*   Chardet

## Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ysen0603/DataScience_app
    cd DataScience_app
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application:

1.  **Activate your virtual environment** (if not already active).
2.  **Navigate to the project root directory** (where `main.py` is located).
3.  **Run the Streamlit application:**
    ```bash
    streamlit run main.py
    ```

    This command will open the application in your default web browser.

4.  **Interact with the application:**
    *   Use the sidebar navigation to switch between different sections: "Chargement des données" (Data Loading), "Exploration des données" (Data Exploration), "Prétraitement" (Preprocessing), "Groupement" (Grouping), "Entraînement du modèle" (Model Training), and "Sauvegarde du modèle" (Model Saving).
    *   Follow the on-screen instructions to upload data, perform analysis, train models, and save/load them.

## Project Structure

```
.
├── main.py                     # Main Streamlit application entry point
├── data/                       # Directory for sample datasets
│   └── Titanic.csv             # Example dataset
├── PG/                         # Contains modules for different stages of the data pipeline
│   ├── __init__.py
│   ├── data_loading.py         # Handles data upload and initial display
│   ├── data_exploration.py     # Provides tools for data visualization and statistics
│   ├── data_preprocessing.py   # Manages data cleaning and transformation
│   ├── data_grouping.py        # Enables data aggregation and grouping
│   ├── model_training.py       # Facilitates machine learning model training
│   └── model_saving.py         # Manages saving and loading of trained models
├── utils/                      # Contains utility functions
│   ├── data_utils.py           # Helper functions for data handling
│   ├── model_utils.py          # Helper functions for model training and evaluation
│   └── preprocessing_utils.py  # Helper functions for data preprocessing
└── requirements.txt            # List of Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## Contact

Developed with ❤️ by Ennaya Yassine yassineennaya@gmail.com