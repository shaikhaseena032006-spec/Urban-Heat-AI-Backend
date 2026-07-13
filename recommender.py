def classify_risk(temp):

    if temp < 38:
        return "Low"

    elif temp < 42:
        return "Medium"

    else:
        return "High"


def recommend_action(ndvi, ndbi, temp):

    if temp > 45 and ndbi > 0.2:
        return "Cool Roof Program"

    elif temp > 45 and ndvi < 0.2:
        return "Mass Tree Plantation"

    elif temp > 42:
        return "Urban Greening"

    else:
        return "Monitor Only"