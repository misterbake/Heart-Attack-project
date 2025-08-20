import pandas as pd
import numpy as np

class Predictor:
    def __init__(self, model):
        self.model = model

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # убираем лишние колонки
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")

        # gender → binary
        def magic(series):
            return (
                series.astype(str)
                .str.lower()
                .replace({"male": 1, "female": 0, "1.0": 1, "0.0": 0})
                .infer_objects(copy=False)
                .astype("int8")
            )

        if "Gender" in df.columns:
            df["Gender"] = magic(df["Gender"])

        # категориальные с заполнением модой
        cols = [
            "Diabetes",
            "Family History",
            "Smoking",
            "Obesity",
            "Alcohol Consumption",
            "Previous Heart Problems",
            "Medication Use",
            "Stress Level",
            "Physical Activity Days Per Week",
        ]
        for col in cols:
            if col in df.columns:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val).astype("int8")

        # heart rate → клиппинг, а не фильтрация
        if "Heart rate" in df.columns:
            q_high = df["Heart rate"].quantile(0.99)
            df["Heart rate"] = df["Heart rate"].clip(upper=q_high)

        # diet → int8
        if "Diet" in df.columns:
            df["Diet"] = df["Diet"].astype("int8")

        # убираем ненужные колонки
        df = df.drop(
            columns=["id", "Sleep Hours Per Day", "Smoking", "Systolic blood pressure"],
            errors="ignore",
        )

        return df

    def predict(self, df: pd.DataFrame):
        X = self._preprocess(df)
        preds = self.model.predict(X)
        return preds
