from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()
model = joblib.load('nigeria_house_price_model.pkl')
columns = joblib.load('model_columns.pkl')

class PropertyInput(BaseModel):
    bedrooms: int
    bathrooms: int
    toilets: int
    parking_space: int
    state: str
    title: str

@app.post("/predict")
def predict(property: PropertyInput):
    new_property = pd.DataFrame(columns=columns)
    new_property.loc[0] = 0
    new_property['bedrooms'] = property.bedrooms
    new_property['bathrooms'] = property.bathrooms
    new_property['toilets'] = property.toilets
    new_property['parking_space'] = property.parking_space

    state_col = f"state_{property.state}"
    title_col = f"title_{property.title}"
    if state_col in new_property.columns:
        new_property[state_col] = True
    if title_col in new_property.columns:
        new_property[title_col] = True

    pred_log = model.predict(new_property)
    pred_price = float(np.expm1(pred_log)[0])
    return {"predicted_price": pred_price}