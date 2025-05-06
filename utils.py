import pandas as pd
import os, urllib.request

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

# Define paths to datasets
def get_data_paths():
    return {
        "raw_data": "data/bikes_paris.csv",
        "clean_data": "data/bike_df_cleaned.csv"
    }

def fetch_from_gcs(local_path: str, bucket: str = "pydaling-assets") -> str:
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        url = f"https://storage.googleapis.com/{bucket}/{local_path}"
        urllib.request.urlretrieve(url, local_path)
    return local_path