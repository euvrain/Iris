# 🌸 Iris Flower Classifier (KNN)

Welcome to the **Iris Flower Classifier**, a beginner-friendly machine learning web app built from scratch using pure Python and Streamlit! 🐍✨

This app uses the classic [Iris dataset]([https://archive.ics.uci.edu/ml/datasets/iris](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)) to predict the species of an iris flower based on 4 input features:  
**sepal length**, **sepal width**, **petal length**, and **petal width**.

> 💡 Built by [@tay](https://github.com/euvrain) to learn the core ML workflow — from dataset → custom KNN model → interactive web app → deployment!

---

## 🔮 Live Demo
Try it here → [irisflowerknn.streamlit.app](https://irisflowerknn.streamlit.app/) 🌼  
Predict flowers. Meet the cat. Vibe responsibly.

---

## 💻 Tech Stack

- 🐍 **Python**
- 📊 **Pandas & NumPy** (for data wrangling & math)
- ✋ **KNN from scratch** (no scikit-learn!)
- 🎨 **Streamlit** (for building the web app)
- 🐱 **GIF** (for mascot ofc)

---

## 🧠 How It Works

1. Loads the Iris dataset
2. Maps species to numeric labels
3. Accepts custom flower measurements from user input
4. Runs a **custom K-Nearest Neighbors (KNN)** algorithm to:
   - Measure distance to each flower in the dataset
   - Find the *k* closest ones
   - Predict the most common species among them
5. Displays the prediction + a lil' cat friend

---

## 🐾 Features

- 🎚 Interactive sliders for inputting flower features
- 🧠 ML prediction logic built from scratch (no libraries!)
- 🐱 Adorable mascot animation next to the title

