import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import gdown
import os

st.set_page_config(page_title="Airbnb Cold Start Rating Prediction", layout="wide")
st.title("Airbnb Cold Start Rating Prediction")

st.markdown("""
Upload a CSV file of Airbnb listings without reviews.  
This tool uses a trained Random Forest model to classify them as **Great**, **Average**, or **Poor**.
""")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# 自动从 Google Drive 下载模型
@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=1yPmTtt-O8whqqLBr8_bzmmS8Cf5s7I2a"
    output = "rf_model.pkl"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return joblib.load(output)

model = load_model()

with st.expander("Project Introduction"):
    st.markdown("""
This project addresses Airbnb's cold start problem by predicting likely ratings for new listings.  
The model uses listing features like `room_type`, `host status`, and `amenities` to predict whether a new listing  
is likely to be rated **Great**, **Average**, or **Poor** by future guests.
""")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.markdown("### Uploaded Preview (Top 5 Rows)")
        st.dataframe(df.head())

        # Align with model input
        expected_cols = list(model.feature_names_in_)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

        # Predict
        predictions = model.predict(df)
        df["Predicted_Rating"] = predictions

        st.markdown(f"**Total Predictions:** {len(df)}")

        # Optional: filter by room_type
        if "room_type" in df.columns:
            selected_room = st.selectbox("Filter by Room Type (optional)", df["room_type"].unique())
            df = df[df["room_type"] == selected_room]

        # Create columns for layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Prediction Results Table")
            st.dataframe(df, use_container_width=True)

            st.markdown("### Rating Proportion Table")
            percent_df = df["Predicted_Rating"].value_counts(normalize=True).reset_index()
            percent_df.columns = ["Rating", "Percentage"]
            percent_df["Percentage"] = (percent_df["Percentage"] * 100).round(2).astype(str) + "%"
            st.dataframe(percent_df, use_container_width=True)

        with col2:
            st.markdown("### Rating Distribution Chart")
            chart_option = st.radio("Select Chart Type", ["Bar", "Horizontal Bar", "Pie"])

            counts = df["Predicted_Rating"].value_counts()
            if chart_option == "Bar":
                st.bar_chart(counts)
            elif chart_option == "Horizontal Bar":
                fig, ax = plt.subplots()
                ax.barh(counts.index, counts.values, color="skyblue")
                ax.set_xlabel("Count")
                ax.set_ylabel("Rating")
                st.pyplot(fig)
            elif chart_option == "Pie":
                fig, ax = plt.subplots()
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

        # Download predictions
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Prediction CSV", csv, "predicted_airbnb.csv", "text/csv")

        # Feature importance (if supported)
        with st.expander("Feature Importance"):
            if hasattr(model, "feature_importances_"):
                importance_df = pd.DataFrame({
                    "Feature": model.feature_names_in_,
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=False)
                st.dataframe(importance_df.head(10), use_container_width=True)
            else:
                st.info("This model does not support feature importance.")

    except Exception as e:
        st.error("Prediction failed. Please check your CSV formatting.")
        st.exception(e)
