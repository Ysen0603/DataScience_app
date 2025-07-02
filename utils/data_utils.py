import pandas as pd
import chardet

def detect_encoding(file):
    raw_data = file.read(10000)  # Read the first 10000 bytes
    result = chardet.detect(raw_data)
    return result['encoding']

def load_excel_or_csv(file, **kwargs):
    try:
        if file.name.endswith('.xlsx'):
            return pd.read_excel(file, **kwargs)
        elif file.name.endswith('.csv'):
            encoding = detect_encoding(file)
            file.seek(0)  # Reset cursor to the beginning of the file
            return pd.read_csv(file, encoding=encoding, **kwargs)
        else:
            raise ValueError("File format not supported. Use .xlsx or .csv")
    except Exception as e:
        raise IOError(f"Error loading file: {str(e)}")