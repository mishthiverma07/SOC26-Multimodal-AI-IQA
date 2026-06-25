# Logistic Regression: Handwritten Digit Recognition
# Based on Week 2 Multiclass Logistic Regression resource.

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

digits = load_digits()

print("Available keys:", dir(digits))
print("Data shape:", digits.data.shape)
print("Target shape:", digits.target.shape)

plt.gray()
plt.matshow(digits.images[0])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    digits.data,
    digits.target,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

print("Model accuracy:", model.score(X_test, y_test))

print("Prediction for first test sample:", model.predict([X_test[0]]))
print("Actual label:", y_test[0])

y_predicted = model.predict(X_test)

cm = confusion_matrix(y_test, y_predicted)
print("Confusion matrix:")
print(cm)