# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset
# data = {
#     'Age': [22, 25, 47, 52, 46, 56, 48, 55, 60, 62, 30],
#     'Salary': [25000, 32000, 78000, 90000, 68000, 76000, 69000, 80000, 83000, 87000, 40000],
#     'Purchased': [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0]
# }

# Create DataFrame
# df = pd.DataFrame(data)
df = pd.read_csv('data.csv')

# Split the data into training and testing sets
X = df[['Age', 'Salary']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)
