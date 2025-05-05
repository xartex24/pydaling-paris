# import streamlit as st
# import pandas as pd
# import numpy as np
# import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns
# from math import sqrt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
# import os
# import pickle
# import gzip
# import gdown

# ######################################
# ###     Data Loading & Cleanup     ###
# ######################################
# @st.cache_data
# def load_data(file_path: str):
#     df = pd.read_csv(file_path)
#     # Basic renaming
#     rename_dict = {
#         'Identifiant du compteur': 'Cntr_ID',
#         'Nom du compteur': 'Cntr_Name',
#         'Identifiant du site de comptage': 'Cntg_Site_ID',
#         'Nom du site de comptage': 'Cntg_Site_Name',
#         'Comptage horaire': 'Hrly_Cnt',
#         'Date et heure de comptage': 'Cntg_Date_Time',
#         "Date d'installation du site de comptage": 'Cntg_Site_Inst_Date',
#         'Lien vers photo du site de comptage': 'Link_Cntg_Site_Photo',
#         'Coordonnées géographiques': 'Geo_Coord',
#         'Identifiant technique compteur': 'Tech_Cntr_ID',
#         'ID Photos': 'Photo_ID',
#         'test_lien_vers_photos_du_site_de_comptage_': 'Test_Link_Cntg_Site_Photo',
#         'id_photo_1': 'Photo_ID_1',
#         'url_sites': 'Site_URLs',
#         'type_dimage': 'Img_type',
#         'mois_annee_comptage': 'Cntg_Month_Year'
#     }
#     df.rename(columns=rename_dict, inplace=True)
#     df['Cntg_Date_Time'] = pd.to_datetime(df['Cntg_Date_Time'], errors='coerce', utc=True)
#     df.dropna(subset=['Cntg_Date_Time'], inplace=True)
#     df['hour'] = df['Cntg_Date_Time'].dt.hour
#     # Split latitude and longitude
#     df[['latitude','longitude']] = df['Geo_Coord'].str.split(',', expand=True)
#     df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
#     df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
#     # Drop rows missing key data, including Cntr_Name
#     df.dropna(subset=['hour','latitude','longitude','Hrly_Cnt','Cntr_Name'], inplace=True)
#     return df

# #########################################
# ###     Feature Engineering Setup     ###
# #########################################
# def calc_distance_from_center(lat, lon, center=(48.8566, 2.3522)):
#     """
#     Approximate Euclidean distance from the city center.
#     For better accuracy, you may use geopy.distance.geodesic.
#     """
#     return sqrt((lat - center[0])**2 + (lon - center[1])**2)

# def bin_hour(h):
#     """Convert numeric hour into a category."""
#     if 5 <= h < 10:
#         return "Morning"
#     elif 10 <= h < 16:
#         return "Afternoon"
#     elif 16 <= h < 21:
#         return "Evening"
#     else:
#         return "Night"

# ###################################################
# ###     Model Training and Precompute Setup     ###
# ###################################################
# def train_model(df):
#     """
#     Train a Random Forest using location-based features.
#     Features used:
#       - 'Cntr_Name' (categorical),
#       - 'distance_center' (numeric),
#       - 'hour_bin' (categorical),
#       - 'hour' (numeric),
#       - 'Month','Day','Weekday_Number' (numeric),
#       - 'latitude','longitude' (numeric).
#     Returns:
#       - trained model,
#       - list of feature names,
#       - dictionary of label encoders,
#       - scaler,
#       - RF Test R² score,
#       - Feature importance figure.
#     """
#     # Create new features
#     df['distance_center'] = df.apply(lambda row: calc_distance_from_center(row['latitude'], row['longitude']), axis=1)
#     df['hour_bin'] = df['hour'].apply(bin_hour)
#     # Assume Month, Day, Weekday_Number already exist in df
#     features = ['Cntr_Name', 'distance_center', 'hour_bin', 'hour', 'Month', 'Day', 'Weekday_Number', 'latitude', 'longitude']
#     target = 'Hrly_Cnt'
#     df_model = df.dropna(subset=features+[target]).copy()
#     X = df_model[features]
#     y = df_model[target]
#     # Label encode categorical columns
#     cat_cols = ['Cntr_Name', 'hour_bin']
#     encoders = {}
#     for c in cat_cols:
#         le = LabelEncoder()
#         X[c] = le.fit_transform(X[c].astype(str))
#         encoders[c] = le
#     # Train-test split with fixed random state
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     # Scale numeric features (all except categorical)
#     numeric_cols = list(set(X.columns) - set(cat_cols))
#     scaler = StandardScaler()
#     X_train_scaled = X_train.copy()
#     X_test_scaled = X_test.copy()
#     X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
#     X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
#     # Train a Random Forest with tuned hyperparameters and fixed random state
#     rf = RandomForestRegressor(n_estimators=100, max_depth=30, min_samples_split=5, random_state=42, n_jobs=-1)
#     rf.fit(X_train_scaled, y_train)
#     y_pred = rf.predict(X_test_scaled)
#     r2_test = r2_score(y_test, y_pred)
#     # Create feature importance figure
#     fi_series = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
#     fig, ax = plt.subplots(figsize=(6,4))
#     sns.barplot(x=fi_series.values, y=fi_series.index, ax=ax, palette='Blues_d')
#     ax.set_title("RF: Feature Importance")
#     plt.tight_layout()
#     # Ensure the directory exists
#     if not os.path.exists("models"):
#         os.makedirs("models")
#     # Save the trained Random Forest model, scaler, and encoders to a pickle file
    

#     # with open("models/rf_model_predicted.pkl", "wb") as f:
#     #     pickle.dump({
#     #         "model": rf,
#     #         "feature_cols": features,
#     #         "encoders": encoders,
#     #         "scaler": scaler,
#     #         "rf_r2": r2_test
#     #     }, f)


#     with gzip.open("models/rf_model_predicted.pkl.gz", "wb") as f:
#         pickle.dump({
#             "model": rf,
#             "feature_cols": features,
#             "encoders": encoders,
#             "scaler": scaler,
#             "rf_r2": r2_test
#         }, f, protocol=pickle.HIGHEST_PROTOCOL)


#     return rf, features, encoders, scaler, r2_test, fig


# # Precompute predictions for the entire period (2025-2026)
# @st.cache_data(show_spinner=True, hash_funcs={
#     type(None): lambda _: None
# })
# def precompute_predictions(_rf_model, feature_cols, _encoders, _scaler, df, top_10_counters):
#     """
#     Precompute predictions for each top counting point for every date and hour in 2025-2026.
#     Returns a DataFrame with columns: 'Cntr_Name', 'date', 'hour', 'predicted_traffic'.
#     """
#     predictions = []
#     start_date = datetime.date(2025, 1, 1)
#     end_date = datetime.date(2026, 12, 31)
#     all_dates = pd.date_range(start=start_date, end=end_date, freq='D').date
#     for counter in top_10_counters:
#         df_counter = df[df["Cntr_Name"] == counter]
#         if df_counter.empty:
#             continue
#         # Get coordinates from the first occurrence
#         lat = df_counter["latitude"].iloc[0]
#         lon = df_counter["longitude"].iloc[0]
#         dist_center = calc_distance_from_center(lat, lon)
#         for d in all_dates:
#             for h in range(24):
#                 dt_sel = datetime.datetime.combine(d, datetime.time(h))
#                 month_val = dt_sel.month
#                 day_val = dt_sel.day
#                 weekday_num_val = dt_sel.weekday()
#                 hour_bin_val = bin_hour(h)
#                 input_data = {
#                     "Cntr_Name": [counter],
#                     "distance_center": [dist_center],
#                     "hour_bin": [hour_bin_val],
#                     "hour": [h],
#                     "Month": [month_val],
#                     "Day": [day_val],
#                     "Weekday_Number": [weekday_num_val],
#                     "latitude": [lat],
#                     "longitude": [lon]
#                 }
#                 input_df = pd.DataFrame(input_data)
#                 # Label encode categorical columns using stored encoders
#                 for cat_col in ["Cntr_Name", "hour_bin"]:
#                     le = _encoders[cat_col]
#                     input_df[cat_col] = le.transform(input_df[cat_col].astype(str))
#                 cat_cols = ["Cntr_Name", "hour_bin"]
#                 numeric_cols = list(set(feature_cols) - set(cat_cols))
#                 input_df_scaled = input_df.copy()
#                 input_df_scaled[numeric_cols] = _scaler.transform(input_df[numeric_cols])
#                 pred = _rf_model.predict(input_df_scaled)[0]
#                 predictions.append({
#                     "Cntr_Name": counter,
#                     "date": d,
#                     "hour": h,
#                     "predicted_traffic": pred
#                 })
#     pred_df = pd.DataFrame(predictions)
#     return pred_df

# #########################################
# ###     Full App: Train + Predict     ###
# #########################################

# def load_trained_model(model_path: str):
#     with gzip.open(model_path, "rb") as f:
#         data = pickle.load(f)
#     return data["model"], data["feature_cols"], data["encoders"], data["scaler"], data["rf_r2"]

# def prediction_page():
#     st.header("Prediction of bicycle traffic")

#     st.markdown("""
#             The generated data below, which is shown by selecting parameters such as date and time,
#             is based on an already trained Random Forest model from the Data Modeling section of this application.
#             """)
    
#     # Load raw data
#     df = load_data("data/bike_df_cleaned.csv")  # Adjust path as needed

#     model_file = "models/rf_model_predicted.pkl.gz"
#     if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
#         # Load the pre-trained model and objects from the pickle file, including rf_r2
#         rf_model, feature_cols, encoders, scaler, rf_r2 = load_trained_model(model_file)
#     else:
#         st.error("Pre-trained model not found. Please run the training page to generate it.")
#         return
        
#     # Get top 10 busiest counters by total traffic
#     top_10_counters = (
#         df.groupby("Cntr_Name")["Hrly_Cnt"]
#         .sum()
#         .sort_values(ascending=False)
#         .head(10)
#         .index.tolist()
#     )

#     with st.spinner("Please wait, we're currently looking into the future!"):
#         # Check if a precomputed file exists; if so, load it; otherwise, precompute and save to CSV.
#         precomp_file = "data/pred_lookup.csv"
#     if os.path.exists(precomp_file) and os.path.getsize(precomp_file) > 0:
#         pred_lookup = pd.read_csv(precomp_file, parse_dates=["date"])
#         # Ensure 'date' is converted to Python date objects
#         pred_lookup["date"] = pred_lookup["date"].dt.date
#     else:
#         pred_lookup = precompute_predictions(rf_model, feature_cols, encoders, scaler, df, top_10_counters)
#         pred_lookup.to_csv(precomp_file, index=False)

#     st.subheader("Select parameters")
#     selected_counter = st.selectbox("Select a counting point (top 10 by total traffic)", top_10_counters)

#     # Filter data for the selected counter and get coordinates
#     df_counter = df[df["Cntr_Name"] == selected_counter].copy()
#     lat = df_counter["latitude"].iloc[0]
#     lon = df_counter["longitude"].iloc[0]

#     # Date & Hour input
#     col1, col2 = st.columns(2)
#     with col1:
#         min_allowed_date = datetime.date(2025, 1, 1)
#         max_allowed_date = datetime.date(2026, 12, 31)
#         selected_date = st.date_input("Select Date (2025-2026)", value=min_allowed_date,
#                                       min_value=min_allowed_date, max_value=max_allowed_date)
#     with col2:
#         selected_hour = st.slider("Select Hour", 0, 23, 12)

#     # Lookup the precomputed prediction for the selected parameters
#     lookup_row = pred_lookup[
#         (pred_lookup["Cntr_Name"] == selected_counter) &
#         (pred_lookup["date"] == selected_date) &
#         (pred_lookup["hour"] == selected_hour)
#     ]
#     if not lookup_row.empty:
#         pred_traffic = lookup_row["predicted_traffic"].values[0]
#     else:
#         pred_traffic = None

#     # Retrieve actual measured traffic (if available) for the selected counter, date, and hour
#     df["date_only"] = df["Cntg_Date_Time"].dt.date
#     # Replace the year with 2024 for the actual data lookup
#     selected_date_2024 = selected_date.replace(year=2024)
#     df_filtered = df[(df["Cntr_Name"] == selected_counter) &
#                     (df["date_only"] == selected_date_2024) &
#                     (df["hour"] == selected_hour)]
#     if not df_filtered.empty:
#         real_val = int(df_filtered["Hrly_Cnt"].mean())
#     else:
#         real_val = None


#     # st.success("Trained model and predictions ready!")
    
#     # Display predicted traffic
#     if pred_traffic is not None:
#         st.markdown(
#             f"""
#             <p style='font-size:24px; font-weight:bold;'>
#                 Predicted traffic:
#                 <span style='color:green;'>{pred_traffic:.2f}</span>
#                 cyclists per hour
#             </p>
#             """,
#             unsafe_allow_html=True
#         )
#     # Display real measured traffic with updated text (and no decimals)
#     if real_val is not None:
#         st.markdown(
#             f"""
#             <p style='font-size:24px; font-weight:bold;'>
#                 Counted in 2024:
#                 <span style='color:blue;'>{real_val}</span>
#                 cyclists per hour
#             </p>
#             """,
#             unsafe_allow_html=True
#         )
    
#     # # Display RF Test R² score and Feature Importance chart below the results
#     # st.markdown(
#     #     f"""
#     #     <p style='font-size:24px; font-weight:bold;'>
#     #         Random Forest R² score: <span style='color:purple;'>{rf_r2:.3f}</span>
#     #     </p>
#     #     """,
#     #     unsafe_allow_html=True
#     # )
#     # # st.pyplot(fi_fig)

# if __name__ == "__main__":
#     prediction_page()







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

##########################################################
###   Google Drive file download and caching feature   ###
##########################################################

@st.cache_data(show_spinner=False)
def download_model_from_drive(drive_url: str, dest_path: str) -> str:
    """
    Download a file from Google Drive only once; subsequent calls use cached file.
    """
    if not os.path.exists(dest_path):
        # Convert shareable URL to direct download URL
        file_id = drive_url.split('/d/')[1].split('/')[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(download_url, dest_path, quiet=False)
    return dest_path

#############################################
###   Compressed model loading function   ###
#############################################

def load_trained_model_from_drive(drive_url: str, local_path: str):
    """
    Download (if needed) and load the trained model from a gzipped pickle.
    """
    model_file = download_model_from_drive(drive_url, local_path)
    with gzip.open(model_file, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_cols"], data["encoders"], data["scaler"], data["rf_r2"]

##################################
###   Data Loading & Cleanup   ###
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
        'Date et heure de comptage': 'Cntg_Date_Time',
        "Date d'installation du site de comptage": 'Cntg_Site_Inst_Date',
        'Lien vers photo du site de comptage': 'Link_Cntg_Site_Photo',
        'Coordonnées géographiques': 'Geo_Coord',
        'Identifiant technique compteur': 'Tech_Cntr_ID',
        'ID Photos': 'Photo_ID',
        'test_lien_vers_photos_du_site_de_comptage_': 'Test_Link_Cntg_Site_Photo',
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
    # Drop rows missing key data, including Cntr_Name
    df.dropna(subset=['hour','latitude','longitude','Hrly_Cnt','Cntr_Name'], inplace=True)
    return df

#####################################
###   Feature Engineering Setup   ###
#####################################

def calc_distance_from_center(lat, lon, center=(48.8566, 2.3522)):
    """
    Approximate Euclidean distance from the city center.
    For better accuracy, you may use geopy.distance.geodesic.
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
###     Model Training and Precompute Setup     ###
###################################################

def train_model(df):
    """
    Train a Random Forest using location-based features.
    Features used:
      - 'Cntr_Name' (categorical),
      - 'distance_center' (numeric),
      - 'hour_bin' (categorical),
      - 'hour' (numeric),
      - 'Month','Day','Weekday_Number' (numeric),
      - 'latitude','longitude' (numeric).
    Returns:
      - trained model,
      - list of feature names,
      - dictionary of label encoders,
      - scaler,
      - RF Test R² score,
      - Feature importance figure.
    """
    # Create new features
    df['distance_center'] = df.apply(lambda row: calc_distance_from_center(row['latitude'], row['longitude']), axis=1)
    df['hour_bin'] = df['hour'].apply(bin_hour)
    # Assume Month, Day, Weekday_Number already exist in df
    features = ['Cntr_Name', 'distance_center', 'hour_bin', 'hour', 'Month', 'Day', 'Weekday_Number', 'latitude', 'longitude']
    target = 'Hrly_Cnt'
    df_model = df.dropna(subset=features + [target]).copy()
    X = df_model[features]
    y = df_model[target]
    # Label encode categorical columns
    cat_cols = ['Cntr_Name', 'hour_bin']
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le
    # Train-test split with fixed random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale numeric features (all except categorical)
    numeric_cols = list(set(X.columns) - set(cat_cols))
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    # Train a Random Forest with tuned hyperparameters and fixed random state
    rf = RandomForestRegressor(n_estimators=100, max_depth=30, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    r2_test = r2_score(y_test, y_pred)
    # Create feature importance figure
    fi_series = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=fi_series.values, y=fi_series.index, ax=ax, palette='Blues_d')
    ax.set_title("RF: Feature Importance")
    plt.tight_layout()
    # Ensure the directory exists
    if not os.path.exists("models"):
        os.makedirs("models")
    # Save the trained Random Forest model, scaler, and encoders to a pickle file
    with gzip.open("models/rf_model_predicted.pkl.gz", "wb") as f:
        pickle.dump({
            "model": rf,
            "feature_cols": features,
            "encoders": encoders,
            "scaler": scaler,
            "rf_r2": r2_test
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    return rf, features, encoders, scaler, r2_test, fig

# Precompute predictions for the entire period (2025-2026)
@st.cache_data(show_spinner=True, hash_funcs={type(None): lambda _: None})
def precompute_predictions(_rf_model, feature_cols, _encoders, _scaler, df, top_10_counters):
    """
    Precompute predictions for each top counting point for every date and hour in 2025-2026.
    Returns a DataFrame with columns: 'Cntr_Name', 'date', 'hour', 'predicted_traffic'.
    """
    predictions = []
    start_date = datetime.date(2025, 1, 1)
    end_date = datetime.date(2026, 12, 31)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D').date
    for counter in top_10_counters:
        df_counter = df[df["Cntr_Name"] == counter]
        if df_counter.empty:
            continue
        # Get coordinates from the first occurrence
        lat = df_counter["latitude"].iloc[0]
        lon = df_counter["longitude"].iloc[0]
        dist_center = calc_distance_from_center(lat, lon)
        for d in all_dates:
            for h in range(24):
                dt_sel = datetime.datetime.combine(d, datetime.time(h))
                month_val = dt_sel.month
                day_val = dt_sel.day
                weekday_num_val = dt_sel.weekday()
                hour_bin_val = bin_hour(h)
                input_data = {
                    "Cntr_Name": [counter],
                    "distance_center": [dist_center],
                    "hour_bin": [hour_bin_val],
                    "hour": [h],
                    "Month": [month_val],
                    "Day": [day_val],
                    "Weekday_Number": [weekday_num_val],
                    "latitude": [lat],
                    "longitude": [lon]
                }
                input_df = pd.DataFrame(input_data)
                # Label encode categorical columns using stored encoders
                for cat_col in ["Cntr_Name", "hour_bin"]:
                    le = _encoders[cat_col]
                    input_df[cat_col] = le.transform(input_df[cat_col].astype(str))
                cat_cols = ["Cntr_Name", "hour_bin"]
                numeric_cols = list(set(feature_cols) - set(cat_cols))
                input_df_scaled = input_df.copy()
                input_df_scaled[numeric_cols] = _scaler.transform(input_df[numeric_cols])
                pred = _rf_model.predict(input_df_scaled)[0]
                predictions.append({
                    "Cntr_Name": counter,
                    "date": d,
                    "hour": h,
                    "predicted_traffic": pred
                })
    return pd.DataFrame(predictions)

#########################################
###     Full App: Train + Predict     ###
#########################################

def prediction_page():
    st.header("Prediction of bicycle traffic")

    # Load raw data
    df = load_data("data/bike_df_cleaned.csv")

    # Google Drive link and local cache path
    DRIVE_LINK = "https://drive.google.com/file/d/1Xsr3agNy-n89ZfZLseirQusB-er4YlLa"
    LOCAL_MODEL_PATH = "models/rf_model_predicted.pkl"

    # Load or download+load the model
    rf_model, feature_cols, encoders, scaler, rf_r2 = load_trained_model_from_drive(
        DRIVE_LINK, LOCAL_MODEL_PATH
    )

    # Get top 10 busiest counters by total traffic
    top_10_counters = (
        df.groupby("Cntr_Name")["Hrly_Cnt"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    with st.spinner("Please wait, we're currently looking into the future!"):
        precomp_file = "data/pred_lookup.csv"
        if os.path.exists(precomp_file) and os.path.getsize(precomp_file) > 0:
            pred_lookup = pd.read_csv(precomp_file, parse_dates=["date"])
            pred_lookup["date"] = pred_lookup["date"].dt.date
        else:
            pred_lookup = precompute_predictions(rf_model, feature_cols, encoders, scaler, df, top_10_counters)
            pred_lookup.to_csv(precomp_file, index=False)

    st.subheader("Select parameters")
    selected_counter = st.selectbox("Select a counting point (top 10 by total traffic)", top_10_counters)

    # Filter and retrieve coordinates
    df_counter = df[df["Cntr_Name"] == selected_counter].copy()
    lat = df_counter["latitude"].iloc[0]
    lon = df_counter["longitude"].iloc[0]

    # Date & Hour input
    col1, col2 = st.columns(2)
    with col1:
        min_allowed_date = datetime.date(2025, 1, 1)
        max_allowed_date = datetime.date(2026, 12, 31)
        selected_date = st.date_input("Select Date (2025-2026)", value=min_allowed_date,
                                      min_value=min_allowed_date, max_value=max_allowed_date)
    with col2:
        selected_hour = st.slider("Select Hour", 0, 23, 12)

    # Lookup the precomputed prediction
    lookup_row = pred_lookup[
        (pred_lookup["Cntr_Name"] == selected_counter) &
        (pred_lookup["date"] == selected_date) &
        (pred_lookup["hour"] == selected_hour)
    ]
    pred_traffic = lookup_row["predicted_traffic"].values[0] if not lookup_row.empty else None

    # Retrieve actual measured traffic (if available)
    df["date_only"] = df["Cntg_Date_Time"].dt.date
    selected_date_2024 = selected_date.replace(year=2024)
    df_filtered = df[
        (df["Cntr_Name"] == selected_counter) &
        (df["date_only"] == selected_date_2024) &
        (df["hour"] == selected_hour)
    ]
    real_val = int(df_filtered["Hrly_Cnt"].mean()) if not df_filtered.empty else None

    # Display predicted traffic
    if pred_traffic is not None:
        st.markdown(
            f"""
            <p style='font-size:24px; font-weight:bold;'>
                Predicted traffic:
                <span style='color:green;'>{pred_traffic:.2f}</span>
                cyclists per hour
            </p>
            """,
            unsafe_allow_html=True
        )

    # Display real measured traffic
    if real_val is not None:
        st.markdown(
            f"""
            <p style='font-size:24px; font-weight:bold;'>
                Counted in 2024:
                <span style='color:blue;'>{real_val}</span>
                cyclists per hour
            </p>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    prediction_page()
