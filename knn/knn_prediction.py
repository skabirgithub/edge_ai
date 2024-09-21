# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from CSV
df = pd.read_csv('fruits.csv')
print(df)

# Separate features (X) and target (y)
X = df[['weight', 'texture', 'color']]  # Features: weight, texture, color
y = df['label']  # Target: label (1 for apple, 0 for orange)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create the KNN classifier (k=3)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Example of predicting a new fruit
# new_fruit = [[155, 1, 1]]  # weight=155, texture=smooth (1), color=red (1)
new_fruit = pd.DataFrame([[155, 1, 1]], columns=['weight', 'texture', 'color'])
prediction = knn.predict(new_fruit)
print(f"Predicted fruit: {'Apple' if prediction[0] == 1 else 'Orange'}")
