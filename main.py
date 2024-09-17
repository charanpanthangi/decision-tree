# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'your_data.csv' with your dataset)
df = pd.read_csv('your_data.csv')

# Display first few rows
print(df.head())

# Univariate analysis - distribution of a single feature
def univariate_analysis(column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Univariate Analysis of {column}')
    plt.show()

# Bivariate analysis - relationship between two features
def bivariate_analysis(x_col, y_col):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[x_col], y=df[y_col])
    plt.title(f'Bivariate Analysis between {x_col} and {y_col}')
    plt.show()

# Example: Univariate analysis on 'feature1' and bivariate analysis between 'feature1' and 'target'
univariate_analysis('feature1')
bivariate_analysis('feature1', 'target')

# Define features (X) and target variable (y) for classification/regression
X = df[['feature1', 'feature2', 'feature3']]  # Input features
y_classification = df['target_class']  # Target for classification
y_regression = df['target_regression']  # Target for regression

# Train-test split for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# Train-test split for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Decision Tree Classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train_class, y_train_class)

# Predict and evaluate classifier
y_pred_class = classifier.predict(X_test_class)
print("Classification Accuracy:", accuracy_score(y_test_class, y_pred_class))
print(classification_report(y_test_class, y_pred_class))

# Save classifier model as pkl file
with open('decision_tree_classifier.pkl', 'wb') as file:
    pickle.dump(classifier, file)

# Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train_reg, y_train_reg)

# Predict and evaluate regressor
y_pred_reg = regressor.predict(X_test_reg)
print("Regression Mean Squared Error:", mean_squared_error(y_test_reg, y_pred_reg))

# Save regressor model as pkl file
with open('decision_tree_regressor.pkl', 'wb') as file:
    pickle.dump(regressor, file)

# Loading and using the saved models

# Load the classifier model
with open('decision_tree_classifier.pkl', 'rb') as file:
    loaded_classifier = pickle.load(file)

# Load the regressor model
with open('decision_tree_regressor.pkl', 'rb') as file:
    loaded_regressor = pickle.load(file)

# Example input for classifier and regressor
new_input = np.array([[1.5, 2.5, 3.0]])  # Replace with actual values

# Classifier prediction
class_prediction = loaded_classifier.predict(new_input)
print(f'Class Prediction: {class_prediction[0]}')

# Regressor prediction
regression_prediction = loaded_regressor.predict(new_input)
print(f'Regression Prediction: {regression_prediction[0]}')
