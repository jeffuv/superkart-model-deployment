import os
import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860").rstrip("/")
st.set_page_config(page_title="SuperKart Sales Prediction", page_icon="🛒")
st.title("SuperKart Sales Prediction")
st.caption("Online and batch inference interface")

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

with tab_single:
    with st.form("single_prediction"):
        col1, col2 = st.columns(2)
        with col1:
            product_weight = st.number_input("Product Weight", min_value=0.01, value=12.66)
            sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
            allocated_area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027, format="%.3f")
            mrp = st.number_input("Product MRP", min_value=0.01, value=117.08)
            product_family = st.selectbox("Product ID Character", ["FD", "DR", "NC"])
        with col2:
            store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
            city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
            store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
            store_age = st.number_input("Store Age in Years", min_value=0, value=17)
            category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])
        submitted = st.form_submit_button("Predict Sales")

    if submitted:
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_family,
            "Store_Age_Years": store_age,
            "Product_Type_Category": category
        }
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
            response.raise_for_status()
            st.metric("Predicted Sales", f"{response.json()['Sales']:,.2f}")
        except requests.RequestException as error:
            st.error(f"Prediction request failed: {error}")

with tab_batch:
    uploaded_file = st.file_uploader("Upload batch CSV", type=["csv"])
    if uploaded_file is not None:
        preview = pd.read_csv(uploaded_file)
        st.dataframe(preview.head(10), use_container_width=True)
        if st.button("Run Batch Prediction"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            try:
                response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files, timeout=60)
                response.raise_for_status()
                result = pd.DataFrame(response.json()["predictions"])
                st.dataframe(result, use_container_width=True)
                st.download_button("Download Predictions", result.to_csv(index=False), "superkart_predictions.csv", "text/csv")
            except requests.RequestException as error:
                st.error(f"Batch prediction request failed: {error}")
