"""
SuperKart Sales Forecasting Dashboard - Streamlit Frontend
"""

import os
import io
import json
import requests
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="SuperKart Sales Forecasting Hub",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .kpi-card { background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); border-radius: 12px; padding: 1.2rem; border-left: 5px solid #2563EB; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1E40AF; }
    .metric-label { font-size: 0.9rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://img.icons8.com/clouds/200/shopping-cart.png", width=120)
st.sidebar.title("Configuration")
default_backend_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:7860")
backend_url = st.sidebar.text_input("Backend API Root URL", value=default_backend_url)
backend_url = backend_url.rstrip("/")

if st.sidebar.button("Test Backend Connection"):
    try:
        res = requests.get(f"{backend_url}/health", timeout=5)
        if res.status_code == 200:
            st.sidebar.success(f"Connected! {res.json().get('service', 'API Online')}")
        else:
            st.sidebar.warning(f"Backend status: {res.status_code}")
    except Exception as e:
        st.sidebar.error(f"Cannot connect to backend: {e}")

@st.cache_resource
def load_local_model():
    candidates = ["superkart_model.joblib", "backend_files/superkart_model.joblib", "../superkart_model.joblib"]
    for path in candidates:
        if os.path.exists(path):
            try: return joblib.load(path)
            except Exception: pass
    return None

local_pipeline = load_local_model()

st.markdown('<div class="main-header">🛒 SuperKart Sales Forecasting Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Enterprise Demand & Revenue Forecasting Solution</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Single Product Prediction", "📁 Batch Inference (CSV)", "📊 Model Performance & Insights"])

with tab1:
    st.subheader("Predict Revenue for an Individual Product & Store")
    with st.form("single_predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📦 Product Attributes")
            product_weight = st.number_input("Product Weight (kg)", min_value=1.0, max_value=35.0, value=12.66, step=0.1)
            product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"], index=0)
            product_allocated_area = st.number_input("Allocated Display Area Ratio", min_value=0.001, max_value=0.350, value=0.0270, step=0.001, format="%.4f")
            product_mrp = st.number_input("Maximum Retail Price (Product MRP) ($)", min_value=10.0, max_value=400.0, value=117.08, step=1.0)
            product_id_char = st.selectbox("Product Category Code", ["FD", "DR", "NC"], index=0)
            product_type_category = st.selectbox("Shelf Life Category", ["Perishables", "Non Perishables"], index=1)
        with col2:
            st.markdown("#### 🏬 Store Attributes")
            store_size = st.selectbox("Store Size", ["Small", "Medium", "High"], index=1)
            store_city_type = st.selectbox("Store City Tier", ["Tier 1", "Tier 2", "Tier 3"], index=1)
            store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Departmental Store", "Food Mart"], index=1)
            store_age = st.slider("Store Age (Years Operating)", min_value=1, max_value=45, value=16)

        submit_btn = st.form_submit_button("🚀 Generate Forecast", use_container_width=True)

    if submit_btn:
        payload = {
            "Product_Weight": float(product_weight),
            "Product_Sugar_Content": product_sugar,
            "Product_Allocated_Area": float(product_allocated_area),
            "Product_MRP": float(product_mrp),
            "Store_Size": store_size,
            "Store_Location_City_Type": store_city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": int(store_age),
            "Product_Type_Category": product_type_category
        }
        pred_val = None
        try:
            res = requests.post(f"{backend_url}/v1/predict", json=payload, timeout=5)
            if res.status_code == 200:
                pred_val = res.json().get("prediction")
        except Exception: pass

        if pred_val is None and local_pipeline is not None:
            pred_val = float(round(local_pipeline.predict(pd.DataFrame([payload]))[0], 2))

        if pred_val is not None:
            st.markdown("---")
            k1, k2, k3 = st.columns(3)
            with k1: st.markdown(f'<div class="kpi-card"><div class="metric-label">Predicted Sales</div><div class="metric-value">${pred_val:,.2f}</div></div>', unsafe_allow_html=True)
            with k2: 
                units = int(pred_val / product_mrp) if product_mrp > 0 else 0
                st.markdown(f'<div class="kpi-card"><div class="metric-label">Estimated Volume</div><div class="metric-value">{units:,} units</div></div>', unsafe_allow_html=True)
            with k3:
                tier = "High Velocity" if pred_val > 3500 else ("Moderate" if pred_val > 2000 else "Low Volume")
                st.markdown(f'<div class="kpi-card"><div class="metric-label">Sales Tier</div><div class="metric-value">{tier}</div></div>', unsafe_allow_html=True)
        else:
            st.error("Prediction failed. Ensure backend service or local model is available.")

with tab2:
    st.subheader("Batch Sales Forecasting via CSV Upload")
    uploaded_file = st.file_uploader("Upload Batch CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} records:")
            st.dataframe(batch_df.head(5), use_container_width=True)
            if st.button("⚡ Run Batch Forecast"):
                batch_preds = None
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    res = requests.post(f"{backend_url}/v1/predictbatch", files=files, timeout=15)
                    if res.status_code == 200:
                        pred_json = res.json()
                        batch_preds = [pred_json[str(i)] for i in range(len(batch_df))]
                except Exception: pass

                if batch_preds is None and local_pipeline is not None:
                    batch_preds = [float(round(p, 2)) for p in local_pipeline.predict(batch_df)]

                if batch_preds is not None:
                    res_df = batch_df.copy()
                    res_df["Predicted_Sales_Total"] = batch_preds
                    st.success(f"Forecasted {len(res_df)} items!")
                    m1, m2 = st.columns(2)
                    m1.metric("Total Batch Revenue", f"${res_df['Predicted_Sales_Total'].sum():,.2f}")
                    m2.metric("Average Revenue / Item", f"${res_df['Predicted_Sales_Total'].mean():,.2f}")
                    st.dataframe(res_df, use_container_width=True)
                    csv_buf = io.StringIO()
                    res_df.to_csv(csv_buf, index=False)
                    st.download_button("📥 Download Results CSV", data=csv_buf.getvalue(), file_name="SuperKart_Predictions.csv", mime="text/csv")
        except Exception as err:
            st.error(f"Error parsing file: {err}")

with tab3:
    st.subheader("Model Evaluation & Architectural Insights")
    st.markdown("Champion Architecture: **Tuned Random Forest Regressor Pipeline**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test RMSE", "$280.51", "-$5.65 vs Baseline")
    c2.metric("Test MAE", "$212.14")
    c3.metric("Test R²", "93.11%")
