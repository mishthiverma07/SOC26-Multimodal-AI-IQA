# Hyperparameter Tuning using GridSearchCV
# Based on Week 2 Hyperparameter Tuning resource.

import pandas as pd
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.3,
    random_state=42
)

model = svm.SVC(kernel="rbf", C=30, gamma="auto")
model.fit(X_train, y_train)

print("Train-test split score:", model.score(X_test, y_test))

print("Cross validation scores:")
print(cross_val_score(svm.SVC(kernel="linear", C=10, gamma="auto"), iris.data, iris.target, cv=5))
print(cross_val_score(svm.SVC(kernel="rbf", C=10, gamma="auto"), iris.data, iris.target, cv=5))
print(cross_val_score(svm.SVC(kernel="rbf", C=20, gamma="auto"), iris.data, iris.target, cv=5))

clf = GridSearchCV(
    svm.SVC(gamma="auto"),
    {
        "C": [1, 10, 20],
        "kernel": ["rbf", "linear"]
    },
    cv=5,
    return_train_score=False
)

clf.fit(iris.data, iris.target)

results = pd.DataFrame(clf.cv_results_)
print(results[["param_C", "param_kernel", "mean_test_score"]])

print("Best score:", clf.best_score_)
print("Best parameters:", clf.best_params_)