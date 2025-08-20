from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from predictor import Predictor

app = FastAPI(title="Heart Disease Predictor")

# Загружаем модель при старте
model = joblib.load("model.pkl")
predictor = Predictor(model)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # читаем csv
    df = pd.read_csv(file.file)
    
    # делаем предсказание
    predictions = predictor.predict(df)
    
    # формируем ответ
    result = pd.DataFrame({
        "id": df["id"],
        "prediction": predictions
    })

    # сохраняем файл
    result.to_csv("predicted.csv", index=False)
    
    return result.to_dict(orient="records")
