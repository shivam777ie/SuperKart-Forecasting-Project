"""
SuperKart Sales Forecasting API - Flask Backend
"""

import os
import io
import json
import logging
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Define expected features
EXPECTED_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category"
]

# Locate and load serialized model pipeline
MODEL_PATH = os.environ.get("MODEL_PATH", "superkart_model.joblib")
if not os.path.exists(MODEL_PATH):
    if os.path.exists("backend_files/superkart_model.joblib"):
        MODEL_PATH = "backend_files/superkart_model.joblib"
    elif os.path.exists("../superkart_model.joblib"):
        MODEL_PATH = "../superkart_model.joblib"

logger.info(f"Loading SuperKart forecasting model from: {MODEL_PATH}")
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}.")
except Exception as e:
    logger.error(f"Error loading model: {e}")


@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint providing API metadata."""
    return jsonify({
        "status": "healthy",
        "service": "SuperKart Sales Forecasting API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "expected_features": EXPECTED_FEATURES
    }), 200


def validate_record(record):
    """Validate that record contains all required features."""
    return [feat for feat in EXPECTED_FEATURES if feat not in record]


@app.route("/v1/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    """
    Online (single) inference endpoint.
    Accepts JSON payload with product & store attributes.
    Returns predicted total sales.
    """
    global model
    if model is None:
        if os.path.exists("superkart_model.joblib"):
            model = joblib.load("superkart_model.joblib")
        elif os.path.exists("backend_files/superkart_model.joblib"):
            model = joblib.load("backend_files/superkart_model.joblib")
        else:
            return jsonify({"status": "error", "message": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "Empty JSON payload provided"}), 400

        missing = validate_record(data)
        if missing:
            return jsonify({
                "status": "error",
                "message": f"Missing required features: {missing}",
                "expected_features": EXPECTED_FEATURES
            }), 400

        input_df = pd.DataFrame([{col: data[col] for col in EXPECTED_FEATURES}])
        raw_pred = model.predict(input_df)[0]
        prediction_val = float(round(raw_pred, 2))

        return jsonify({
            "status": "success",
            "prediction": prediction_val,
            "currency": "USD",
            "formatted_sales": f"${prediction_val:,.2f}"
        }), 200

    except Exception as e:
        logger.error(f"Error during single prediction: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/v1/predictbatch", methods=["POST"])
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Batch inference endpoint.
    Accepts CSV file upload or JSON list of records.
    Returns JSON dictionary mapping row index to predicted sales.
    """
    global model
    if model is None:
        if os.path.exists("superkart_model.joblib"):
            model = joblib.load("superkart_model.joblib")
        elif os.path.exists("backend_files/superkart_model.joblib"):
            model = joblib.load("backend_files/superkart_model.joblib")
        else:
            return jsonify({"status": "error", "message": "Model not loaded"}), 500

    try:
        df = None
        if "file" in request.files:
            file_obj = request.files["file"]
            if file_obj.filename == "":
                return jsonify({"status": "error", "message": "No file selected"}), 400
            df = pd.read_csv(file_obj)
        elif request.is_json:
            json_data = request.get_json()
            if isinstance(json_data, list):
                df = pd.DataFrame(json_data)
            elif isinstance(json_data, dict) and "records" in json_data:
                df = pd.DataFrame(json_data["records"])
            else:
                return jsonify({"status": "error", "message": "Invalid JSON format for batch"}), 400
        else:
            return jsonify({"status": "error", "message": "No file or JSON data provided"}), 400

        missing = [feat for feat in EXPECTED_FEATURES if feat not in df.columns]
        if missing:
            return jsonify({
                "status": "error",
                "message": f"Batch data is missing required features: {missing}",
                "expected_features": EXPECTED_FEATURES
            }), 400

        features_df = df[EXPECTED_FEATURES]
        preds = model.predict(features_df)
        pred_dict = {str(i): float(round(val, 2)) for i, val in enumerate(preds)}

        return jsonify(pred_dict), 200

    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
