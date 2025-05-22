import pandas as pd
import gdown
import os, urllib.request
from google.cloud import storage
import streamlit as st

def format_dataframe(df):
    """
    Format numeric columns in European number format (1.000,25)
    while preserving specific columns like years
    """
    # Create a copy of the dataframe to avoid modifying the original
    formatted_df = df.copy()
    
    numeric_cols = formatted_df.select_dtypes(include=['float64', 'int64']).columns
    
    
    exclude_cols = ['year', 'Year', 'annee', 'Annee']
    
    # Format numeric columns
    for col in numeric_cols:
        if col in exclude_cols:
            continue
        
        if formatted_df[col].dtype == 'int64':
            formatted_df[col] = formatted_df[col].apply(lambda x: f'{x:,.0f}'.replace(',', '.') if pd.notnull(x) else x)
        
        elif formatted_df[col].dtype == 'float64':
            formatted_df[col] = formatted_df[col].apply(lambda x: 
                f'{x:,.2f}'.replace(',', 'TEMP').replace('.', ',').replace('TEMP', '.') 
                if pd.notnull(x) else x
            )
    
    return formatted_df

# # Define paths to datasets
# def get_data_paths():
#     return {
#         "raw_data": "data/bikes_paris.csv",
#         "clean_data": "data/bike_df_cleaned.csv"
#     }

@st.cache_data(show_spinner=False)
def load_bikes_paris():
    local_path = "data/bikes_paris.csv"
    # if the file does not exist or is a pointer (small), download the real one
    if (not os.path.exists(local_path)
        or os.path.getsize(local_path) < 1000):
        # Google Drive URL
        file_id = "ВАШИЯ_FILE_ID"
        url = f"https://drive.google.com/uc?id={"1PBhJTE_2SH-UOavUHgTVV5nwwb0sOm8m"}"
        gdown.download(url, local_path, quiet=False)
    # the real CSV:
    return pd.read_csv(local_path, sep=";")

# Page loading:
def data_exploration_page():
    df = load_bikes_paris()
    st.write("Rows:", len(df), "Columns:", df.shape[1])
    st.dataframe(df.head())

# def fetch_from_gcs(local_path: str, bucket: str = "pydaling-assets") -> str:
#     if not os.path.exists(local_path):
#         os.makedirs(os.path.dirname(local_path), exist_ok=True)
#         url = f"https://storage.googleapis.com/{bucket}/{local_path}"
#         urllib.request.urlretrieve(url, local_path)
#     return local_path


# def fetch_from_gcs(local_path: str, bucket: str = "pydaling-assets") -> str:
#     if not os.path.exists(local_path):
#         os.makedirs(os.path.dirname(local_path), exist_ok=True)
#         # Сглобяваме пълния път към обекта
#         url = f"https://storage.googleapis.com/{bucket}/{local_path}"
#         print(">>> FETCH URL:", url)         # <-- това ще видиш в Cloud Run логовете
#         urllib.request.urlretrieve(url, local_path)
#     return local_path

def fetch_from_gcs(local_path: str, bucket: str = "pydaling-assets") -> str:
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client = storage.Client()
        blb = client.bucket(bucket).blob(local_path)
        blb.download_to_filename(local_path)
    return local_path