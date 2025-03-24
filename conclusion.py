import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def render_conclusion_page():

    st.title("Conclusion")
    st.write("""
    Our project aimed to identify the best model for predicting the number of cyclists per hour in Paris.
    To do this, we had to evaluate several approaches – including **Linear Regression (LR)**,
    **Random Forest (RF)**, **Decision Tree (DT)**, as well as clustering methods such as **K-Means** and **DBSCAN**.
    \nOn the other hand, K-Means and DBSCAN are **clustering methods, not regression ones**.
    R² estimates the explained variance in a dependent variable, which, as in our case, is not applicable to clustering data
    and for this reason is not mentioned in the graphs below.
    \n\nThe graphs below highlight the key results of our model comparisons.
    """)


    # Create a DataFrame
    data = {
        "Model": ["LR (Geo)", "LR (Non-Geo)", "RF (Geo)", "RF (Non-Geo)", "DT (Geo)", "DT (Non-Geo)"],
        "MAE": [56.59, 56.01, 21.62, 20.81, 24.57, 31.38],
        "RMSE": [92.28, 92.07, 43.5, 44, 50.26, 59.64],
        "Train R²-Score": [0.203, 0.202, 0.875, 0.896, 0.877, 0.902],
        "Test R²-Score": [0.206, 0.203, 0.824, 0.818, 0.764, 0.665],
    }

    df = pd.DataFrame(data)

    
    def highlight_best(s):
        """Highlight the best model row in blue."""
        return ['background-color: #AFCBF5; font-weight: bold' if s.name == 2 else '' for _ in s]

    styled_df = (
        df.style
        .apply(highlight_best, axis=1)  
        .set_properties(**{"text-align": "center"})  
        .format(precision=3) 
    )

    
    st.header("Model Performance Comparison")
    st.dataframe(styled_df, hide_index=True, use_container_width=True)

    
    st.subheader("R² Score Comparison")

    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["Model"], df["Test R²-Score"], color="#1f497d", width=0.6)

    
    for i, v in enumerate(df["Test R²-Score"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=12, fontweight='bold')

    
    ax.set_xlabel("Model")
    ax.set_ylabel("R² Score")
    ax.set_title("R² Score Comparison", fontsize=14)
    ax.set_ylim(0, 0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    
    st.pyplot(fig)

    st.write("""
Our main conclusions from the analysis are as follows:
 - Adding **geographic data** resulted in only minor changes in the model results.
 - The **Random Forest model, when combined with geographic data**, emerged as the most accurate and robust for predicting the number of cyclists per hour.
 - Although the **Decision Tree** shows good performance, it shows some overfitting.
 - The small effect of **longitude and latitude** suggests that temporal factors have a greater influence on the number of bicycles.
 - **Nonlinear models are crucial** for capturing complex patterns in data, which is even more evident from the underperformance of **Linear Regression**.

We also used **HeatMap** visualization to explore cyclist behavior, although should not be the sole basis for decision making.

Ultimately, based on our evaluation, **the Random Forest model with geographic data** was selected as the best model for our case, due to its high R² score (0.824) and consistent performance on unseen data. The analysis also confirmed that factors such as **time of day**, **longitude** and **latitude** were the most important predictors that aligned with the observed patterns.""")