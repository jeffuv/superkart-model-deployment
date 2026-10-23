from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).with_name("superkart_model.joblib")
model = joblib.load(MODEL_PATH)

FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
    "Product_MRP", "Store_Size", "Store_Location_City_Type", "Store_Type",
    "Product_Id_char", "Store_Age_Years", "Product_Type_Category"
]
NUMERIC_FEATURES = {
    "Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"
}

superkart_api = Flask("SuperKart")

def validate_record(record):
    if not isinstance(record, dict):
        raise ValueError("Input must be a JSON object.")
    missing = [column for column in FEATURES if column not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    clean = {column: record[column] for column in FEATURES}
    for column in NUMERIC_FEATURES:
        clean[column] = float(clean[column])
    return clean

@superkart_api.get("/")
def home():
    return jsonify({"message": "Welcome to the SuperKart System", "status": "ready"})

@superkart_api.post("/v1/predict")
def predict_sales():
    try:
        clean = validate_record(request.get_json())
        prediction = float(model.predict(pd.DataFrame([clean], columns=FEATURES))[0])
        return jsonify({"Sales": round(prediction, 2)})
    except Exception as error:
        return jsonify({"error": str(error)}), 400

@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    try:
        if "file" not in request.files:
            raise ValueError("Upload a CSV file using the form field 'file'.")
        input_data = pd.read_csv(request.files["file"])
        missing = [column for column in FEATURES if column not in input_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        clean_data = input_data[FEATURES].copy()
        for column in NUMERIC_FEATURES:
            clean_data[column] = pd.to_numeric(clean_data[column], errors="raise")
        predictions = model.predict(clean_data)
        output = input_data.copy()
        output["Predicted_Product_Store_Sales_Total"] = predictions.round(2)
        return jsonify({"count": len(output), "predictions": output.to_dict(orient="records")})
    except Exception as error:
        return jsonify({"error": str(error)}), 400

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
