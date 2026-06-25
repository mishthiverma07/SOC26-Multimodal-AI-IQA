# Customer Churn Prediction using ANN
# Based on Week 3 resource: Customer Churn Prediction using ANN.

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

# Sample telecom-style churn dataset for practice
df = pd.DataFrame({
    "tenure": [1, 5, 12, 24, 36, 48, 60, 72, 3, 8, 15, 30],
    "MonthlyCharges": [29.8, 56.9, 65.2, 70.5, 89.1, 95.6, 105.3, 110.5, 35.4, 49.9, 68.8, 77.2],
    "TotalCharges": [29.8, 284.5, 782.4, 1692.0, 3207.6, 4588.8, 6318.0, 7956.0, 106.2, 399.2, 1032.0, 2316.0],
    "gender": ["Female", "Male", "Male", "Female", "Male", "Female", "Male", "Female", "Female", "Male", "Female", "Male"],
    "InternetService": ["DSL", "Fiber optic", "DSL", "Fiber optic", "Fiber optic", "DSL", "Fiber optic", "DSL", "DSL", "Fiber optic", "DSL", "Fiber optic"],
    "Churn": [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
})

# Convert categorical columns using one-hot encoding
df = pd.get_dummies(df, columns=["gender", "InternetService"], drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Scale numeric columns
scaler = MinMaxScaler()
X[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.fit_transform(
    X[["tenure", "MonthlyCharges", "TotalCharges"]]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = keras.Sequential([
    keras.layers.Dense(12, input_shape=(X_train.shape[1],), activation="relu"),
    keras.layers.Dense(6, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=50)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

y_predicted = model.predict(X_test)
y_predicted_labels = []

for value in y_predicted:
    if value > 0.5:
        y_predicted_labels.append(1)
    else:
        y_predicted_labels.append(0)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predicted_labels))

print("Classification Report:")
print(classification_report(y_test, y_predicted_labels))