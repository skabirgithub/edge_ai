# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 48, 55, 60, 62],
    'Salary': [25000, 32000, 78000, 90000, 68000, 76000, 69000, 80000, 83000, 87000],
    'Purchased': [0, 0, 1, 1, 1, 1, 0, 1, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Split the data into training and testing sets
X = df[['Age', 'Salary']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# New data for prediction (Age, Salary)
new_data = pd.DataFrame({
    'Age': [30, 40, 50],  # New ages
    'Salary': [40000, 60000, 85000]  # Corresponding salaries
})

# Predict using the trained model
predictions = model.predict(new_data)

# Print the predictions
for i, pred in enumerate(predictions):
    status = "Purchased" if pred == 1 else "Did not purchase"
    print(f"Person {i+1} (Age: {new_data['Age'][i]}, Salary: {new_data['Salary'][i]}) -> {status}")
