# Simple Linear Regression: Home Price Prediction
# Based on Week 2 Simple Linear Regression resource.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.DataFrame({
    "area": [2600, 3000, 3200, 3600, 4000],
    "price": [550000, 565000, 610000, 680000, 725000]
})

plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(df.area, df.price, marker="+")
plt.show()

model = linear_model.LinearRegression()
model.fit(df[["area"]], df.price)

print("Prediction for 3300 sq ft:", model.predict([[3300]]))
print("Prediction for 5000 sq ft:", model.predict([[5000]]))

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)