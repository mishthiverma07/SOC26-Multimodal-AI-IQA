# K-Means Clustering: Age and Income
# Based on Week 2 KMeans clustering resource.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df = pd.DataFrame({
    "Name": [
        "person1", "person2", "person3", "person4", "person5",
        "person6", "person7", "person8", "person9", "person10",
        "person11", "person12", "person13", "person14", "person15"
    ],
    "Age": [27, 29, 29, 28, 42, 39, 41, 38, 36, 35, 37, 26, 27, 28, 29],
    "Income": [70000, 90000, 61000, 60000, 150000, 155000, 160000, 162000, 156000, 130000, 137000, 45000, 48000, 51000, 49500]
})

plt.scatter(df["Age"], df["Income"])
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Age vs Income")
plt.show()

km = KMeans(n_clusters=3, random_state=42, n_init=10)
y_predicted = km.fit_predict(df[["Age", "Income"]])

df["cluster"] = y_predicted
print(df)

# Scaling because age and income have very different ranges
scaler = MinMaxScaler()

df["Income"] = scaler.fit_transform(df[["Income"]])
df["Age"] = scaler.fit_transform(df[["Age"]])

km = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(df[["Age", "Income"]])

print("\nAfter scaling:")
print(df)

df0 = df[df.cluster == 0]
df1 = df[df.cluster == 1]
df2 = df[df.cluster == 2]

plt.scatter(df0.Age, df0.Income)
plt.scatter(df1.Age, df1.Income)
plt.scatter(df2.Age, df2.Income)
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("KMeans Clusters after Scaling")
plt.show()

# Elbow method
sse = []
k_range = range(1, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df[["Age", "Income"]])
    sse.append(km.inertia_)

plt.plot(k_range, sse)
plt.xlabel("K")
plt.ylabel("Sum of Squared Error")
plt.title("Elbow Method")
plt.show()