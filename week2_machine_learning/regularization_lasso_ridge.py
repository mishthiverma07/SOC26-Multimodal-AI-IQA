# L1 and L2 Regularization Practice
# Based on Week 2 Regularization resource.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge

df = pd.DataFrame({
    "area": [2600, 3000, 3200, 3600, 4000, 4100, 4500, 5000, 5500, 6000],
    "bedrooms": [3, 4, 3, 3, 5, 4, 5, 5, 6, 6],
    "age": [20, 15, 18, 30, 8, 10, 5, 4, 3, 2],
    "price": [550000, 565000, 610000, 595000, 760000, 810000, 850000, 900000, 950000, 1000000]
})

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

print("Linear Regression train score:", linear_model.score(X_train, y_train))
print("Linear Regression test score:", linear_model.score(X_test, y_test))

lasso_model = Lasso(alpha=50, max_iter=10000)
lasso_model.fit(X_train, y_train)

print("\nLasso Regression train score:", lasso_model.score(X_train, y_train))
print("Lasso Regression test score:", lasso_model.score(X_test, y_test))

ridge_model = Ridge(alpha=50)
ridge_model.fit(X_train, y_train)

print("\nRidge Regression train score:", ridge_model.score(X_train, y_train))
print("Ridge Regression test score:", ridge_model.score(X_test, y_test))