from predictor import predict_temperature


def simulate_scenario(
    ndvi,
    ndbi,
    ndwi,
    lat,
    lon,
    ndvi_increase=0,
    ndbi_decrease=0,
    ndwi_increase=0
):

    current_temp = predict_temperature(
        ndvi,
        ndbi,
        ndwi,
        lat,
        lon
    )

    future_ndvi = ndvi + ndvi_increase
    future_ndbi = max(0, ndbi - ndbi_decrease)
    future_ndwi = ndwi + ndwi_increase

    future_temp = predict_temperature(
        future_ndvi,
        future_ndbi,
        future_ndwi,
        lat,
        lon
    )

    cooling_effect = round(
        current_temp - future_temp,
        2
    )

    return {
        "current_temp": current_temp,
        "future_temp": future_temp,
        "cooling_effect": cooling_effect
    }