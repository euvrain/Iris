import numpy as np
import pandas as pd
import streamlit as st

#Load the dataset
df = pd.read_csv('IRIS.csv')

#Convert species to numbers
species2num = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
num2species = {v: k for k, v in species2num.items()}

df['target'] = df['species'].map(species2num)

#Features and labels
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df['target'].values

# KNN from scratch
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, x_test, k=3):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))
    distances.sort()
    neighbors = [label for _, label in distances[:k]]
    return max(set(neighbors), key=neighbors.count)

# Streamlit app
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🌸 Iris Flower Classifier (KNN)")

with col2:
    st.image("https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif", width=100)

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

if st.button("Predict Flower Type"):
    new_flower = np.array([sepal_length, sepal_width, petal_length, petal_width])
    prediction = knn_predict(X, y, new_flower)
    st.success(f"🌼 Predicted: **{num2species[prediction]}**")


