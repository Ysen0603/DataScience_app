import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import numpy as np

def encode_categorical(df, columns, method):
    if method == "One-Hot Encoding":
        return pd.get_dummies(df, columns=columns)
    elif method == "Label Encoding":
        le = LabelEncoder()
        for col in columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def normalize_data(df, columns, method):
    if method == "StandardScaler":
        scaler = StandardScaler()
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
    
    df[columns] = scaler.fit_transform(df[columns])
    return df

def handle_missing_values(df, columns, method):
    if method == "Delete":
        return df.dropna(subset=columns)
    elif method == "Mean":
        imputer = SimpleImputer(strategy='mean')
    elif method == "Median":
        imputer = SimpleImputer(strategy='median')
    elif method == "Most Frequent Value":
        imputer = SimpleImputer(strategy='most_frequent')
    
    df[columns] = imputer.fit_transform(df[columns])
    return df

def detect_column_types(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    return numeric_columns, categorical_columns

def handle_outliers(df, columns, method='IQR'):
    for col in columns:
        if method == 'IQR':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
    return df

def encode_high_cardinality(df, columns, max_categories=10):
    for col in columns:
        if df[col].nunique() > max_categories:
            top_categories = df[col].value_counts().nlargest(max_categories).index
            df[col] = df[col].where(df[col].isin(top_categories), 'Other')
    return df