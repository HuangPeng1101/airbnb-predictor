import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import gdown
import os

st.set_page_config(page_title="Airbnb Cold Start Rating Prediction", layout="wide")
st.title("🏠 Airbnb Cold Start Rating Prediction")

st.markdown("""
Welcome to the **Airbnb Cold Start Tool**!  
Upload a CSV file of Airbnb listings **without reviews**, and let our machine learning model predict whether your listing would be considered:
- 🌟 **Great**
- 😐 **Average**
- 👎 **Poor**

---  
""")

# Sidebar upload
with st.sidebar:
    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader("Upload your Airbnb CSV file", type=["csv"])
    st.markdown("👉 File size < 200MB")

# Load model from Google Drive
@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=1HVDUDq74DsL5hMRwcL9bFBK9wgJOvgZ-"  # ← 改成你的ID
    output = "rf_model.pkl"
    if not os.path.exists(output):
        st.info("📥 Downloading model...")
        gdown.download(url, output, quiet=False)
        st.success("✅ Model downloaded!")
    try:
        return joblib.load(output)
    except Exception as e:
        st.error("❌ Failed to load model.")
        st.exception(e)
        st.stop()

model = load_model()

with st.expander("📘 Project Introduction"):
    st.markdown("""
This project addresses Airbnb's cold start problem by predicting **likely ratings** for new listings.  
We use listing features like `room_type`, `host status`, and `amenities` to estimate outcomes **without any reviews**.
""")

# If file uploaded
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Align with model input
        expected_cols = list(model.feature_names_in_)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

        # Predict
        predictions = model.predict(df)
        df["Predicted_Rating"] = predictions

        st.success(f"✅ Predictions completed for {len(df)} listings!")

        # Sidebar filtering
        with st.sidebar:
            if "room_type" in df.columns:
                selected_room = st.selectbox("Filter by Room Type", df["room_type"].unique())
                df = df[df["room_type"] == selected_room]
            if "neighbourhood_group" in df.columns:
                selected_group = st.selectbox("Group by Neighbourhood (Optional)", df["neighbourhood_group"].unique())
                group_df = df[df["neighbourhood_group"] == selected_group]
            else:
                selected_group = None
                group_df = df.copy()

        # Layout for charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📋 Prediction Results Table")
            st.dataframe(group_df, use_container_width=True)

            st.markdown("### 📈 Rating Proportion")
            percent_df = group_df["Predicted_Rating"].value_counts(normalize=True).reset_index()
            percent_df.columns = ["Rating", "Percentage"]
            percent_df["Percentage"] = (percent_df["Percentage"] * 100).round(2).astype(str) + "%"
            st.dataframe(percent_df, use_container_width=True)

        with col2:
            st.markdown("### 📊 Distribution Chart")
            chart_option = st.radio("Chart Type", ["Bar", "Horizontal", "Pie"], horizontal=True)
            counts = group_df["Predicted_Rating"].value_counts()
            if chart_option == "Bar":
                st.bar_chart(counts)
            elif chart_option == "Horizontal":
                fig, ax = plt.subplots()
                ax.barh(counts.index, counts.values, color="lightgreen")
                ax.set_xlabel("Count")
                ax.set_ylabel("Rating")
                st.pyplot(fig)
            elif chart_option == "Pie":
                fig, ax = plt.subplots()
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

        # Grouped mean chart (New Feature)
        if selected_group:
            st.markdown("### 🧠 Group-wise Analysis")
            if "room_type" in df.columns:
                grouped = group_df.groupby("room_type")["Predicted_Rating"].value_counts().unstack(fill_value=0)
                st.bar_chart(grouped)

        # Feature importance
        with st.expander("🧪 Feature Importance"):
            if hasattr(model, "feature_importances_"):
                importance_df = pd.DataFrame({
                    "Feature": model.feature_names_in_,
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=False)
                st.dataframe(importance_df.head(10), use_container_width=True)
            else:
                st.info("Model does not support feature importance.")

        # Download
        csv = group_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions CSV", csv, "predicted_airbnb.csv", "text/csv")

    except Exception as e:
        st.error("Prediction failed. Please check your CSV formatting.")
        st.exception(e)
