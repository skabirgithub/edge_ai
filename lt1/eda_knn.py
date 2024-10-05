# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
file_path = 'Books.csv'  # Path to the dataset file
data = pd.read_csv(file_path)

# ---------------- Exploratory Data Analysis (EDA) ---------------- #
# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Get summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# ---------------- K-Nearest Neighbors (KNN) Prediction ---------------- #
# Preprocess the data: Define features and target variable
X = data[['Price']]  # Example feature, adjust based on your dataset
y = data['Rating']  # Target variable (Replace 'Rating' with actual column name for prediction)

# Convert target to categorical values (since KNN is classification)
y = pd.cut(y, bins=[2.0, 3.0, 4.0, 5.0], labels=[0, 1, 2])  # Categorize ratings into 3 classes, adjust as needed

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k)

# Train the model
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# ---------------- Model Evaluation ---------------- #
# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", accuracy)

# Plot correlation heatmap (only for numeric columns)
numeric_columns = ['Price', 'Rating']  # Only select numeric columns

# Create the first figure for the correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show(block=False)  # Show the figure without blocking the code execution

# Create a second figure for the boxplot of numerical features
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[numeric_columns])
plt.title("Boxplot of Numerical Features")
plt.show()




# ---------------- Insights for Students ---------------- #
# 1. Discuss feature correlations and important variables based on the heatmap.
# 2. Analyze how well the KNN model performs based on classification metrics (precision, recall, F1-score).
# 3. Test the model with different values of 'k' in KNN and compare accuracy.
# 4. Discuss how scaling impacts model performance.

