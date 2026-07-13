import joblib

model = joblib.load(
    "../models/urban_heat_model.pkl"
)

def predict_temperature(
    ndvi,
    ndbi,
    ndwi,
    lat,
    lon
):

    features = [[
        ndvi,
        ndbi,
        ndwi,
        lat,
        lon
    ]]

    prediction = model.predict(features)[0]

    return round(float(prediction), 2)