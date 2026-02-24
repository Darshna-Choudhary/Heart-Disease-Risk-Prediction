from fastapi import FastAPI
import joblib
import numpy as np
from app.schema import PatientData

app = FastAPI()

model = joblib.load("app/model.pkl")

@app.get("/")
def home():
    return {"message": "Heart Disease Risk Prediction API"}

@app.post("/predict")
def predict(data: PatientData):
    input_data = np.array([[ 
        data.male, data.age, data.currentSmoker, data.cigsPerDay,
        data.BPMeds, data.prevalentStroke, data.prevalentHyp,
        data.diabetes, data.totChol, data.sysBP,
        data.diaBP, data.BMI, data.heartRate, data.glucose
    ]])

    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "predicted_risk_probability": float(probability),
        "risk_label": "High Risk" if probability >= 0.4 else "Low Risk"
    }