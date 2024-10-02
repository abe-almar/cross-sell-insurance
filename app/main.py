from fastapi import FastAPI  # type: ignore
from pydantic import BaseModel  # type: ignore
import pickle
import pandas as pd

# Create FastAPI app
app = FastAPI()

# Load the trained model and preprocessing pipeline
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)


# Define the request schema using Pydantic
class PredictionRequest(BaseModel):
    Age: int
    Annual_Premium: float
    Gender: str
    Region_Code: str
    Vehicle_Age: str
    Vehicle_Damage: str
    Policy_Sales_Channel: int
    Previously_Insured: int
    Driving_License: int


# Define a root endpoint
@app.get("/")
def root():
    return {"health_check": "OK", "message": "Cross-sell Insurance Model API"}


# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert the incoming request to a DataFrame
    input_data = pd.DataFrame([request.dict()])

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Return the prediction
    return {"prediction": int(prediction[0])}
