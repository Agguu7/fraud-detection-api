from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Fraud Detection API", description="ML-powered fraud detection by Agguuu7", version="1.0")

with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

class Transaction(BaseModel):
    features: List[float]

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!", "status": "healthy"}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    df = pd.DataFrame([transaction.features])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability) * 100, 2),
        "status": "🚨 FRAUD DETECTED" if prediction == 1 else "✅ LEGITIMATE"
    }
