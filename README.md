# IrisFlowerClassification_DataZenixSolutions
📌 Project Overview

This project is part of my internship at DataZenix, where I worked on building a machine learning model to classify Iris flowers into three species (Setosa, Versicolor, Virginica) using their sepal and petal measurements.

The project demonstrates the complete ML workflow – from EDA to model training, evaluation, and prediction.

📊 Dataset

Source: Iris Dataset (UCI Machine Learning Repository)

Features:

Sepal Length

Sepal Width

Petal Length

Petal Width

Target Variable: Species (Setosa, Versicolor, Virginica)

🔍 Steps Performed
1. Data Loading & Exploration

Loaded dataset using Pandas

Checked missing values, dataset info, and summary statistics

Performed EDA with Seaborn and Matplotlib (pairplots, correlation heatmaps)

2. Data Preprocessing

Feature selection: Dropped the Species column from features (X)

Feature scaling using StandardScaler

3. Model Training

Trained and evaluated multiple models:

Logistic Regression

Random Forest Classifier

K-Nearest Neighbors (KNN)

4. Model Evaluation

Accuracy Score

Classification Report (Precision, Recall, F1-Score)

Confusion Matrix (visualized with heatmaps)

5. Prediction on New Data

Created new sample flower data

Scaled and predicted the species using trained models

🛠️ Tech Stack

Python

Pandas, Numpy

Matplotlib, Seaborn

Scikit-learn (sklearn)

📈 Results

All models achieved high accuracy (>90%)

KNN and Random Forest performed the best

Successfully predicted unseen flower samples

Example output:

Flower [1, 5.3, 3.5, 1.4, 0.2] --> Predicted Species: Iris-setosa  
Flower [2, 6.5, 3.0, 5.2, 2.0] --> Predicted Species: Iris-virginica  

🚀 Key Learnings

Data preprocessing and scaling

Model training and evaluation in classification tasks

Visualizing data and confusion matrices for better interpretation

Building reusable functions for model selection

📌 Future Improvements

Hyperparameter tuning for better performance

Deploying the model with Flask / Streamlit for real-time predictions

Expanding to larger datasets for multi-class classification

🙌 Acknowledgment

Special thanks to DataZenix for the opportunity to work on this project during my internship!
