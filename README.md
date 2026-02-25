❤️ Heart Disease Risk Prediction (End-to-End ML Deployment)
📌 Project Overview

This project predicts the 10-year risk of Coronary Heart Disease (CHD) using the Framingham Heart Study dataset.

The objective was not just to train a model, but to build a complete end-to-end machine learning system, including:

Data preprocessing

Model tuning

Imbalanced classification handling

Threshold optimization

REST API deployment

Frontend integration

Production deployment on Render

🚀 Live Demo

🔹 Frontend (Streamlit UI):
[Add your Streamlit URL here]

🔹 Backend API (FastAPI):
[Add your FastAPI URL here]

🧠 Problem Statement

Early detection of cardiovascular risk is crucial.
This project focuses on building a screening-oriented model that prioritizes recall, reducing false negatives in high-risk patients.

📊 Dataset

Source: Framingham Heart Study

Target Variable: TenYearCHD

Features: 14 clinical & lifestyle parameters including:

Age

Blood Pressure

BMI

Cholesterol

Glucose

Smoking status

Diabetes

Heart rate

etc.

⚙️ ML Pipeline

A production-ready Scikit-learn Pipeline was used:

Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=3000, class_weight="balanced"))
])
Why Logistic Regression?

Interpretable (important for healthcare)

Provides coefficient-level feature insights

Stable baseline model

Works well with standardized medical data

🔍 Model Optimization

Train/Test Split (Stratified)

5-Fold Cross Validation

GridSearchCV for hyperparameter tuning

Evaluation metrics:

ROC-AUC

Precision-Recall Curve

F1 Score

Confusion Matrix

Calibration Curve

📈 Model Performance
Metric	Score
Cross-Validated ROC-AUC	~0.73
Test ROC-AUC	~0.70
F1 Score (Positive Class)	~0.35
Recall (Threshold = 0.4)	~0.77
Threshold Optimization

Instead of using default 0.5 threshold, a lower threshold (0.4) was chosen to:

Increase recall

Reduce false negatives

Suit screening use-case

🏗 System Architecture
User → Streamlit Frontend → FastAPI Backend → ML Pipeline → Prediction
Backend

FastAPI

Uvicorn

Deployed on Render

Frontend

Streamlit

Calls backend via REST API

Deployed as separate Render service

🧪 API Endpoint
POST /predict

Request Body:

{
  "male": 1,
  "age": 45,
  "currentSmoker": 0,
  "cigsPerDay": 0,
  "BPMeds": 0,
  "prevalentStroke": 0,
  "prevalentHyp": 0,
  "diabetes": 0,
  "totChol": 200,
  "sysBP": 120,
  "diaBP": 80,
  "BMI": 25,
  "heartRate": 70,
  "glucose": 90
}

Response:

{
  "predicted_risk_probability": 0.19,
  "risk_label": "Low Risk"
}
🚀 Deployment Details
Backend

Platform: Render

Python Service

Cold-start aware handling

Frontend

Platform: Render

Separate Web Service

Timeout & error handling added

🔒 Production Considerations

Preprocessing pipeline saved using joblib

No data leakage

Class imbalance handled via class_weight="balanced"

Proper exception handling in frontend

Status code validation before JSON parsing

📁 Project Structure
backend/
  ├── app/
  │   ├── main.py
  │   ├── model.pkl
  │   └── schema.py
  ├── requirements.txt
  └── runtime.txt

frontend/
  ├── app.py
  └── requirements.txt
🛠 Tech Stack

Python

Scikit-learn

FastAPI

Streamlit

Uvicorn

Render

GitHub

🎯 Key Learnings

Importance of ML pipelines in production

Handling imbalanced datasets

Threshold tuning for business objectives

Cold-start behavior in cloud deployments

Debugging API errors in live environment

Separation of frontend and backend services

🔮 Future Improvements

Compare with XGBoost / LightGBM

Add SHAP interpretability

Improve calibration

Add model versioning

Add monitoring & logging

👩‍💻 Author

Darshna Choudhary
Machine Learning & Backend Enthusiast
