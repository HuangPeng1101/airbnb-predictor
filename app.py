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
    url = "https://drive.google.com/uc?id=1HVDUDq74DsL5hMRwcL9bFBK9wgJOvgZ-"  # ← 你的模型ID
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

        # --- Sidebar: Choose a label column and value to filter ---
        with st.sidebar:
            st.header("🔖 Optional Filtering")
            label_options = ['room_type', 'city', 'neighborhood', 'host_is_superhost', 'full_time_host']
            available_options = [col for col in label_options if col in df.columns]
            if available_options:
                group_col = st.selectbox("Select a column to filter by", available_options)
                selected_value = st.selectbox(f"Value from '{group_col}'", df[group_col].unique())
                df = df[df[group_col] == selected_value]

        # Layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📋 Prediction Results")
            st.dataframe(df, use_container_width=True)

            st.markdown("### 📈 Rating Proportion")
            percent_df = df["Predicted_Rating"].value_counts(normalize=True).reset_index()
            percent_df.columns = ["Rating", "Percentage"]
            percent_df["Percentage"] = (percent_df["Percentage"] * 100).round(2).astype(str) + "%"
            st.dataframe(percent_df, use_container_width=True)

        with col2:
            st.markdown("### 📊 Rating Distribution")
            chart_option = st.radio("Chart Type", ["Bar", "Horizontal", "Pie"], horizontal=True)
            counts = df["Predicted_Rating"].value_counts()
            if chart_option == "Bar":
                st.bar_chart(counts)
            elif chart_option == "Horizontal":
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

        # Download predictions
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Prediction CSV", csv, "predicted_airbnb.csv", "text/csv")

    except Exception as e:
        st.error("Prediction failed. Please check your CSV formatting.")
        st.exception(e)
