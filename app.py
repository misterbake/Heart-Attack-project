from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib

#загружаю модель
model = joblib.load("model.pkl")

app = FastAPI(title="Heart Risk Predictor")

@app.get("/")
def root():
    return {"message": "API работает"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    #читаю CSV из запроса
    df = pd.read_csv(file.file)
    
    ids = df["id"]
    X = df.drop("id", axis=1)

    #предсказываю
    preds = model.predict(X)

    
    result = pd.DataFrame({
        "id": ids,
        "prediction": preds
    })

    return result.to_dict(orient="records")
