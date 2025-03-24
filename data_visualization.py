import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

# Additional imports for the folium map
import folium
from folium.plugins import HeatMap
from folium import MacroElement
from jinja2 import Template
import streamlit.components.v1 as components

def data_visualization_page():
    
    st.header("📊 Data Visualization")
    st.write("Down below, we can see both the initial temporal and geographical analysis, creating broad overview of the dataset")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📉 Temporal Analysis", "📍 Geographical Analysis"])

    if 'bikes_paris_clean' in st.session_state and st.session_state.bikes_paris_clean is not None:
       
        df = st.session_state.bikes_paris_clean.copy()
        
        # ========================== Hourly Trends & Seasonality Analysis ==========================
        with tab1:
            st.write("In this section we will visualize the temporal data regarding the hourly counts with focus on several different parameters.")

            if "Year" in df.columns:
                df["Year"] = df["Year"].astype(int)
                min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
                
                all_years = st.checkbox("Show all years", value=True)

                if all_years:
                    filtered_df = df.copy()
                    selected_year = "Total"
                else:
                    # Use a slider instead of a dropdown for year selection
                    selected_year = st.slider("Select Year", min_year, max_year, min_year)
                    filtered_df = df[df["Year"] == selected_year]

                # Average Hourly Count per Weekday
                hourly_avg = filtered_df.groupby(["hour", "Weekday"])["Hrly_Cnt"].mean().reset_index()

                # Average Hourly Count by Time of Day
                time_of_day_avg = filtered_df.groupby("Time_Of_Day")["Hrly_Cnt"].mean().reset_index()

                # Weekly Average for Seasonality Trend
                if "Date" in filtered_df.columns:
                    # Ensure Date column is datetime
                    if not pd.api.types.is_datetime64_any_dtype(filtered_df["Date"]):
                        filtered_df["Date"] = pd.to_datetime(filtered_df["Date"])
                    
                    weekly_avg = filtered_df.resample("W", on="Date")["Hrly_Cnt"].mean().reset_index()
                    
                    time_of_day_labels = ["Early morning", "Morning", "Middle of the day", "Afternoon", "Evening", "Night"]

                    # Line chart
                    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

                    # Create the plot with ordered hue
                    fig1 = plt.figure(figsize=(10, 6))
                    sns.lineplot(data=hourly_avg, x="hour", y="Hrly_Cnt", hue="Weekday", hue_order=weekday_order, palette="cubehelix")
                    plt.title(f"Avg. Hourly Bicycle Counts - {selected_year}", fontsize=14)
                    plt.xlabel("Hour", fontsize=12)
                    plt.ylabel("Avg. Count", fontsize=12)
                    plt.legend(fontsize=10, loc="upper right", frameon=False)
                    plt.tight_layout()
                    st.pyplot(fig1)

                    # Bar chart 
                    fig2 = plt.figure(figsize=(10, 6))
                    sns.barplot(data=time_of_day_avg, x="Time_Of_Day", y="Hrly_Cnt", palette="cubehelix")
                    plt.title("Avg. Hourly Count by Time of Day", fontsize=14)
                    plt.xlabel("Time of Day", fontsize=12)
                    plt.ylabel("Avg. Count", fontsize=12)
                    plt.xticks(range(len(time_of_day_labels)), time_of_day_labels, rotation=30, ha="right", fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    # Sample a subset
                    max_points = 20000
                    if len(filtered_df) > max_points:
                        sampled_df = filtered_df.sample(n=max_points, random_state=42)
                    else:
                        sampled_df = filtered_df
                    # Scatter plot
                    fig3 = plt.figure(figsize=(12, 7))  
                    sns.scatterplot(
                        data=sampled_df, 
                        x="Time_Of_Day", 
                        y="Hrly_Cnt", 
                        hue="IsWeekend_name", 
                        alpha=0.5, 
                        palette="cubehelix", 
                        s=300
                    )
                    plt.title("Weekend vs. Weekday Distribution", fontsize=16, pad=20)
                    plt.xlabel("Time of Day", fontsize=13, labelpad=15)
                    plt.ylabel("Hourly Count", fontsize=13, labelpad=15)
                    unique_ticks = sorted(sampled_df['Time_Of_Day'].unique())
                    # If the number of ticks does not match the expected labels, adjust accordingly
                    if len(unique_ticks) == len(time_of_day_labels):
                        tick_labels = time_of_day_labels
                    else:
                        # Create a mapping for existing ticks if possible, or simply use the tick values as labels
                        tick_labels = [time_of_day_labels[i] if i < len(time_of_day_labels) else str(i) for i in range(len(unique_ticks))]
                    plt.xticks(
                        unique_ticks,
                        tick_labels,
                        rotation=45,
                        ha="right",
                        fontsize=11
                    )
                    plt.subplots_adjust(bottom=0.15)
                    plt.legend(fontsize=11, loc="upper right", framealpha=0.9, title="Day Type")
                    plt.tight_layout()
                    st.pyplot(fig3)                    


                    # Line chart for weekly average
                    fig4 = plt.figure(figsize=(10, 6))
                    sns.lineplot(data=weekly_avg, x="Date", y="Hrly_Cnt", palette="cubehelix")
                    plt.fill_between(weekly_avg["Date"], weekly_avg["Hrly_Cnt"], alpha=0.2, color="gray") 
                    plt.title("Average Weekly Bicycle Count", fontsize=14)
                    plt.xlabel("Date", fontsize=12)
                    plt.ylabel("Count", fontsize=12)
                    plt.xticks(rotation=30)
                    plt.tight_layout()
                    st.pyplot(fig4) 

                    # Observations
                    st.markdown("""
                    ### Key Insights:
                    - **High peaks in hourly biking counts** can be seen around hour 7 and 17, indicating work traffic.
                    - **Hourly counts distributed by time of day** further confirm this trend.
                    - **Weekends show more spread-out usage**, likely due to leisure or tourism.
                    - **Seasonal trends peak in summer and decline in winter.**
                    """)
                else:
                    st.error("Date column not found in the dataset")
            else:
                st.error("Year column not found in the dataset")

        # ========================== Geographical Analysis ==========================
        with tab2:
            st.write("In this section we will visualize the geographical data, comparing multiple counter sites and displaying them on a map.")

            if "Cntr_Name" in df.columns and "Hrly_Cnt" in df.columns:
                location_counts_df = df.groupby("Cntr_Name")["Hrly_Cnt"].sum().reset_index()
                location_counts_df.columns = ["Location", "Total_Hourly_Count"]

                location_counts_df["Total_Hourly_Count"].fillna(0, inplace=True)
                location_counts_df = location_counts_df[location_counts_df["Total_Hourly_Count"] > 0]

                top_10_locations = location_counts_df.sort_values(by="Total_Hourly_Count", ascending=False).drop_duplicates().head(10)

                # Показваме заглавието като subheader над бар-плота
                st.subheader("Top 10 Locations by Hourly Counts")

                # Set up Figure for horizontal bar chart
                fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
                sns.barplot(
                    data=top_10_locations, 
                    x="Total_Hourly_Count", 
                    y="Location", 
                    palette="cubehelix", 
                    ax=ax
                )
                # Махаме/коментираме ax.set_title(...) за да не се повтаря заглавието
                # ax.set_title("Top 10 Locations by Hourly Counts", fontsize=10)
                ax.yaxis.set_label_position("right") 
                ax.yaxis.tick_right()
                ax.set_xlabel("Sum of Hourly Counts", fontsize=10)
                ax.set_ylabel("Location", fontsize=10)
                ax.tick_params(axis="both", labelsize=6)
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                sns.despine(left=True, bottom=True)

                # Save the figure as PNG and display it
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
                buf.seek(0)
                st.image(buf, use_container_width=True)
            else:
                st.error("Required columns (Cntr_Name, Hrly_Cnt) not found in the dataset")

            # -------------------- New HeatMap Visualization --------------------
            # Caching the bike data to prevent reloading on every page load
            @st.cache_data
            def load_bike_data():
                # Load the CSV
                df_cleaned = pd.read_csv("data/bike_df_cleaned.csv")
                # Drop rows with missing longitude/latitude
                df_cleaned.dropna(subset=["longitude", "latitude"], inplace=True)
                return df_cleaned
            
            bike_df = load_bike_data()
            
            # Подзаглавие за HeatMap
            st.subheader("HeatMap visualization")
            
            # Prepare data for HeatMap (longitude first, then latitude)
            heat_data = bike_df[["longitude", "latitude"]].values.tolist()
            
            # Define custom gradient using string keys
            custom_gradient = {str(k): v for k, v in {
                0.83: "lightblue",
                0.86: "skyblue",
                0.89: "dodgerblue",
                0.92: "blue",
                0.95: "purple",
                0.97: "crimson",
                1.0: "darkred"
            }.items()}
            
            # Create the base folium map centered on Paris
            map_paris = folium.Map(
                location=[48.8566, 2.3522],
                zoom_start=12,
                control_scale=True,
                tiles="CartoDB positron"
            )
            
            # Add the heatmap layer with specified parameters
            HeatMap(
                heat_data,
                radius=12,
                blur=5,
                min_opacity=0.8,
                gradient=custom_gradient,
                name="HeatMap"
            ).add_to(map_paris)
            
            # Define the landmarks with markers
            landmarks = [
                {'name': 'Eiffel Tower', 'coords': [48.8584, 2.2945],
                 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg'},
                {'name': 'Louvre Museum', 'coords': [48.8606, 2.3376],
                 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/6/66/Louvre_Museum_Wikimedia_Commons.jpg'},
                {'name': 'Notre-Dame Cathedral', 'coords': [48.8530, 2.3499],
                 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/f/f7/Notre-Dame_de_Paris%2C_4_October_2017.jpg'},
                {'name': 'Arc de Triomphe', 'coords': [48.8738, 2.2950],
                 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/7/79/Arc_de_Triomphe%2C_Paris_21_October_2010.jpg'},
                {'name': 'Sacré-Cœur Basilica', 'coords': [48.8867, 2.3431],
                 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/c/c5/Le_sacre_coeur.jpg'},
                {'name': 'Place de la Concorde', 'coords': [48.8656, 2.3211],
                 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/f/fa/Place_de_la_Concorde_from_the_Eiffel_Tower%2C_Paris_April_2011.jpg'}
            ]
            
            landmark_layer = folium.FeatureGroup(name="Landmarks").add_to(map_paris)
            
            # Add markers for each landmark with a popup showing an image
            for landmark in landmarks:
                popup_html = f"""
                <div style="text-align:center;">
                    <strong>{landmark['name']}</strong><br>
                    <img src="{landmark['img_url']}" width="150">
                </div>
                """
                folium.Marker(
                    location=landmark['coords'],
                    tooltip=landmark['name'],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color="green", icon="star", icon_color="white")
                ).add_to(landmark_layer)
            
            # Allow toggling layers
            folium.LayerControl().add_to(map_paris)

            # Custom MacroElement for finer zoom increments
            class ZoomOptions(MacroElement):
                def __init__(self, zoomDelta=0.2, zoomSnap=0.2):
                    super(ZoomOptions, self).__init__()
                    self._template = Template(u"""
                    {% macro script(this, kwargs) %}
                        {{ this._parent.get_name() }}.options.zoomDelta = {{ this.zoomDelta }};
                        {{ this._parent.get_name() }}.options.zoomSnap = {{ this.zoomSnap }};
                    {% endmacro %}
                    """)
                    self.zoomDelta = zoomDelta
                    self.zoomSnap = zoomSnap
            
            # Add custom zoom options to the map
            map_paris.add_child(ZoomOptions(zoomDelta=0.2, zoomSnap=0.2))
            
            # Prevent reloading of the map on zoom changes by caching the generated HTML map
            if "map_paris_html" not in st.session_state:
                st.session_state["map_paris_html"] = map_paris._repr_html_()
            
            # Display the folium map using streamlit components
            components.html(st.session_state["map_paris_html"], height=400, width=700)
            
            st.markdown("""
            ### Key Insights:
            - **Bike usage is concentrated in key locations**, with top areas having nearly twice the counts of lower-ranked locations.
            - **Traffic flow is primarily South-North and North-South**, influenced by commuting patterns and tourists moving toward the city center.
            - **High-traffic areas could benefit from improved cycling infrastructure**, such as expanded bike lanes, bike-sharing programs, or safety enhancements.
            """)
    else:
        st.warning("Cleaned dataset is not available. Please check file paths and data loading process.")
