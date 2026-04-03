from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# --- 1️⃣ Load the model ---
MODEL_FILE = "random_forest_heart_disease_model.pkl"
best_model = joblib.load(MODEL_FILE)

# --- 2️⃣ Define normalization ranges ---
estimated_original_ranges = {
    'age': {'min': 15, 'max': 77},
    'sex': {'min': 0, 'max': 1},
    'resting_bp': {'min': 75, 'max': 200},
    'max_heart_rate': {'min': 71, 'max': 202},
    'chest_pain_type': {'min': 0, 'max': 3},
    'resting_ecg': {'min': 0, 'max': 2},
    'st_depression': {'min': 0, 'max': 6.2},
    'st_slope': {'min': 1, 'max': 3},
    'exercise_angina': {'min': 0, 'max': 1},
    'num_major_vessels': {'min': 0, 'max': 3},
    'thalassemia': {'min': 1, 'max': 3}
}

def normalize_value(value, feature_name):
    ranges = estimated_original_ranges[feature_name]
    normalized = (value - ranges['min']) / (ranges['max'] - ranges['min'])
    normalized = (normalized - 0.5) * 4
    return normalized

def predict_heart_disease_real(real_patient_data):
    normalized_data = {feature: normalize_value(value, feature) 
                       for feature, value in real_patient_data.items()}
    patient_df = pd.DataFrame([normalized_data])
    prediction = best_model.predict(patient_df)[0]
    probability = best_model.predict_proba(patient_df)[0][1]
    return {
        "prediction": "YES" if prediction == 1 else "NO",
        "probability": float(probability),
        "normalized_data": normalized_data
    }

# --- 3️⃣ FastAPI app ---
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict heart disease probability for a single patient",
    version="1.0"
)

# --- 4️⃣ Enable CORS ---
origins = [
    "*"  # Allow all origins for testing. Replace with specific domain in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5️⃣ Pydantic model for request body ---
class PatientData(BaseModel):
    age: float
    sex: int
    resting_bp: float
    max_heart_rate: float
    chest_pain_type: int
    resting_ecg: int
    st_depression: float
    st_slope: int
    exercise_angina: int
    num_major_vessels: int
    thalassemia: int

# --- 6️⃣ Prediction endpoint ---
@app.post("/predict")
def predict(patient: PatientData):
    patient_dict = patient.dict()
    result = predict_heart_disease_real(patient_dict)
    return result
