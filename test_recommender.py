from recommender import classify_risk
from recommender import recommend_action

temp = 43.09

print(classify_risk(temp))

print(
    recommend_action(
        0.15,
        0.30,
        temp
    )
)