import joblib

try:
    model = joblib.load("../models/urban_heat_model.pkl")
except Exception:
    model = None


def predict_temperature(
    ndvi,
    ndbi,
    ndwi,
    lat,
    lon
):

    if model is None:
        return 45.2

    features = [[
        ndvi,
        ndbi,
        ndwi,
        lat,
        lon
    ]]

    prediction = model.predict(features)[0]

    return round(float(prediction), 2)