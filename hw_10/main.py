import joblib
import uvicorn

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

with open("elastic_model.pkl", 'rb') as file:
    model = joblib.load(file)

features = ["total_square", "rooms", "floor", "lat", "lon"]

class ModelRequestData(BaseModel):
    total_square: float
    rooms: float
    floor: float
    lat: float
    lon: float

class Result(BaseModel):
    result: float


def make_prediction(data: dict) -> float:
    """ Функция для получения предсказания моделью"""
    input_df = pd.DataFrame([data], columns=features)
    result = model.predict(input_df)[0]
    return float(result)

@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)


@app.get("/predict_get", response_model=Result)
def predict_get(
        total_square: float,
        rooms: float,
        floor: float,
        lat: float,
        lon: float,
):
    data = {
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
        "lat": lat,
        "lon": lon,
    }
    result = make_prediction(data)
    return Result(result=result)


@app.post("/predict_post", response_model=Result)
def predict_post(data: ModelRequestData):
    input_data = data.dict()
    result = make_prediction(input_data)
    return Result(result=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
