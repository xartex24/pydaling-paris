import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os
import pickle
import gzip
import gdown
from utils import fetch_from_gcs

##########################################################
# Google Drive file download and caching feature
##########################################################


local_csv = fetch_from_gcs("data/bike_df_cleaned.csv")
df = pd.read_csv(local_csv)

local_model = fetch_from_gcs("models/rf_model_predicted.pkl.gz")
with gzip.open(local_model,"rb") as f:
    data = pickle.load(f)


@st.cache_data(show_spinner=False)
def download_model_from_drive(drive_url: str, dest_path: str) -> str:
    """
    Download a file from Google Drive only once; subsequent calls use cached file.
    """
    if not os.path.exists(dest_path):
        file_id = drive_url.split('/d/')[1].split('/')[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(download_url, dest_path, quiet=False)
    return dest_path

##########################################################
# Load compressed (gzip) pickle model from Drive
##########################################################

def load_gzip_model_from_drive(drive_url: str, local_path: str):
    """
    Download (if needed) and load the trained model from a gzipped pickle.
    Returns: model, feature_cols, encoders, scaler, rf_r2
    """
    model_file = download_model_from_drive(drive_url, local_path)
    with gzip.open(model_file, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_cols"], data["encoders"], data["scaler"], data["rf_r2"]

##################################
# Data Loading & Cleanup
##################################

@st.cache_data
def load_data(file_path: str):
    df = pd.read_csv(file_path)
    # Basic renaming
    rename_dict = {
        'Identifiant du compteur': 'Cntr_ID',
        'Nom du compteur': 'Cntr_Name',
        'Identifiant du site de comptage': 'Cntg_Site_ID',
        'Nom du site de comptage': 'Cntg_Site_Name',
        'Comptage horaire': 'Hrly_Cnt',
        'Date et heure на comptage': 'Cntg_Date_Time',
        "Date d'installation du site de comptage": 'Cntg_Site_Inst_Date',
        'Lien към фото': 'Link_Cntg_Site_Photo',
        'Coordonnées géographiques': 'Geo_Coord',
        'Identifiant technique compteur': 'Tech_Cntr_ID',
        'ID Photos': 'Photo_ID',
        'id_photo_1': 'Photo_ID_1',
        'url_sites': 'Site_URLs',
        'type_dimage': 'Img_type',
        'mois_annee_comptage': 'Cntg_Month_Year'
    }
    df.rename(columns=rename_dict, inplace=True)
    df['Cntg_Date_Time'] = pd.to_datetime(df['Cntg_Date_Time'], errors='coerce', utc=True)
    df.dropna(subset=['Cntg_Date_Time'], inplace=True)
    df['hour'] = df['Cntg_Date_Time'].dt.hour
    # Split latitude and longitude
    df[['latitude','longitude']] = df['Geo_Coord'].str.split(',', expand=True)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df.dropna(subset=['hour','latitude','longitude','Hrly_Cnt','Cntr_Name'], inplace=True)
    return df

#####################################
# Feature Engineering Setup
#####################################

def calc_distance_from_center(lat, lon, center=(48.8566, 2.3522)):
    """
    Approximate Euclidean distance from the city center.
    """
    return sqrt((lat - center[0])**2 + (lon - center[1])**2)

def bin_hour(h):
    """Convert numeric hour into a category."""
    if 5 <= h < 10:
        return "Morning"
    elif 10 <= h < 16:
        return "Afternoon"
    elif 16 <= h < 21:
        return "Evening"
    else:
        return "Night"

###################################################
# Model Training and Precompute Setup
###################################################

def train_model(df):
    """
    Train a Random Forest using location-based features and save compressed model+metadata.
    """
    df['distance_center'] = df.apply(lambda r: calc_distance_from_center(r['latitude'], r['longitude']), axis=1)
    df['hour_bin'] = df['hour'].apply(bin_hour)
    features = ['Cntr_Name','distance_center','hour_bin','hour','Month','Day','Weekday_Number','latitude','longitude']
    target = 'Hrly_Cnt'
    df_model = df.dropna(subset=features+[target]).copy()
    X, y = df_model[features], df_model[target]
    # Label encode categoricals
    cat_cols = ['Cntr_Name','hour_bin']
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    numeric_cols = list(set(features) - set(cat_cols))
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled [numeric_cols] = scaler.transform(X_test[numeric_cols])
    rf = RandomForestRegressor(n_estimators=100, max_depth=30, min_samples_split=5,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    r2_test = r2_score(y_test, y_pred)
    # Feature importance plot
    fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=fi.values, y=fi.index, ax=ax, palette='Blues_d')
    ax.set_title("RF: Feature Importance")
    plt.tight_layout()
    # Save compressed model+metadata
    os.makedirs("models", exist_ok=True)
    with gzip.open("models/rf_model_predicted.pkl.gz","wb") as f:
        pickle.dump({
            "model": rf,
            "feature_cols": features,
            "encoders": encoders,
            "scaler": scaler,
            "rf_r2": r2_test
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    return rf, features, encoders, scaler, r2_test, fig

@st.cache_data(show_spinner=True, hash_funcs={type(None): lambda _: None})
def precompute_predictions(rf_model, feature_cols, encoders, scaler, df, top_10):
    """
    Precompute daily/hourly predictions for given counters.
    """
    preds = []
    start, end = datetime.date(2025,1,1), datetime.date(2026,12,31)
    all_dates = pd.date_range(start=start, end=end, freq='D').date
    for counter in top_10:
        df_ct = df[df['Cntr_Name']==counter]
        if df_ct.empty: continue
        lat, lon = df_ct[['latitude','longitude']].iloc[0]
        dist = calc_distance_from_center(lat, lon)
        for d in all_dates:
            for h in range(24):
                dt = datetime.datetime.combine(d, datetime.time(h))
                row = {
                    "Cntr_Name": counter,
                    "distance_center": dist,
                    "hour_bin": bin_hour(h),
                    "hour": h,
                    "Month": dt.month,
                    "Day": dt.day,
                    "Weekday_Number": dt.weekday(),
                    "latitude": lat,
                    "longitude": lon
                }
                in_df = pd.DataFrame([row])
                for c in ["Cntr_Name","hour_bin"]:
                    in_df[c] = encoders[c].transform(in_df[c].astype(str))
                num_cols = list(set(feature_cols)-{"Cntr_Name","hour_bin"})
                in_df[num_cols] = scaler.transform(in_df[num_cols])
                preds.append({
                    "Cntr_Name": counter,
                    "date": d,
                    "hour": h,
                    "predicted_traffic": rf_model.predict(in_df)[0]
                })
    return pd.DataFrame(preds)

#########################################
# Full App: Train + Predict
#########################################

def prediction_page():
    st.header("Prediction of bicycle traffic")

    # Load raw data
    df = load_data("data/bike_df_cleaned.csv")

    # Google Drive link and local cache path for the compressed model
    DRIVE_LINK       = "https://drive.google.com/file/d/1bqHJrxODJ87Ec-JMSKpKBtv-BfyX020v/view?usp=sharing"
    LOCAL_MODEL_PATH = "models/rf_model_predicted.pkl.gz"

    # Download & load the model + metadata
    rf_model, feature_cols, encoders, scaler, rf_r2 = load_gzip_model_from_drive(
        DRIVE_LINK, LOCAL_MODEL_PATH
    )

    # Prepare top 10 counters
    top_10 = (
        df.groupby("Cntr_Name")["Hrly_Cnt"]
          .sum()
          .nlargest(10)
          .index
          .tolist()
    )

    # Precompute or load lookup table
    precomp_file = "data/pred_lookup.csv"
    if os.path.exists(precomp_file) and os.path.getsize(precomp_file) > 0:
        pred_lookup = pd.read_csv(precomp_file, parse_dates=["date"])
        pred_lookup["date"] = pred_lookup["date"].dt.date
    else:
        pred_lookup = precompute_predictions(rf_model, feature_cols, encoders, scaler, df, top_10)
        pred_lookup.to_csv(precomp_file, index=False)

    # UI controls
    st.subheader("Select parameters")
    counter    = st.selectbox("Top 10 counters", top_10)
    col1, col2 = st.columns(2)
    with col1:
        sel_date = st.date_input("Select Date (2025-2026)",
                                 datetime.date(2025,1,1),
                                 min_value=datetime.date(2025,1,1),
                                 max_value=datetime.date(2026,12,31))
    with col2:
        sel_hour = st.slider("Select Hour", 0, 23, 12)

    # Lookup prediction
    row = pred_lookup[
        (pred_lookup["Cntr_Name"]==counter) &
        (pred_lookup["date"]==sel_date) &
        (pred_lookup["hour"]==sel_hour)
    ]
    pred_val = row["predicted_traffic"].iloc[0] if not row.empty else None

    # Real measured traffic (2024)
    df["date_only"] = df["Cntg_Date_Time"].dt.date
    real_df = df[
        (df["Cntr_Name"]==counter) &
        (df["date_only"]==sel_date.replace(year=2024)) &
        (df["hour"]==sel_hour)
    ]
    real_val = int(real_df["Hrly_Cnt"].mean()) if not real_df.empty else None

    # Display results
    if pred_val is not None:
        st.markdown(
            f"<p style='font-size:24px;'><b>Predicted:</b> <span style='color:green;'>{pred_val:.2f}</span></p>",
            unsafe_allow_html=True
        )
    if real_val is not None:
        st.markdown(
            f"<p style='font-size:24px;'><b>Counted in 2024:</b> <span style='color:blue;'>{real_val}</span></p>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    prediction_page()
