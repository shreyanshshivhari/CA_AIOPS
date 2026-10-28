from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

# Create FastAPI application
app = FastAPI(title="Iris Flower Prediction API")

# Load trained model
saved_data = joblib.load("model.joblib")

model = saved_data["model"]
target_names = saved_data["target_names"]

# Define input data
class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Iris ML API is running"
    }

# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

# Prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)[0]

    return {
        "predicted_class": int(prediction),
        "predicted_species": target_names[prediction]
    }