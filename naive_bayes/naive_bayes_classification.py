# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset from CSV
df = pd.read_csv('fruits.csv')

# Separate features (X) and target (y)
X = df[['weight', 'texture', 'color']]  # Features: weight, texture, color
y = df['label']  # Target: label (Apple or Orange)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = nb_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Example of predicting a new fruit
new_fruit = pd.DataFrame([[155, 1, 1]], columns=['weight', 'texture', 'color'])  # weight=155, texture=smooth, color=red
prediction = nb_classifier.predict(new_fruit)
print(f"Predicted fruit: {prediction[0]}")
