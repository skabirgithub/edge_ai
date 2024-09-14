# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import messagebox

# Load the dataset from a CSV file
df = pd.read_csv('data.csv')  # Path to your CSV file

# Split the data into training and testing sets
X = df[['Age', 'Salary']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Function to predict purchase based on Age and Salary input


def predict_purchase():
    age = int(entry_age.get())  # Get Age input from the user
    salary = int(entry_salary.get())  # Get Salary input from the user

    # Prepare data for prediction
    new_data = pd.DataFrame({'Age': [age], 'Salary': [salary]})
    prediction = model.predict(new_data)[0]  # Predict based on input

    # Display result in a message box
    status = "Purchased" if prediction == 1 else "Not purchase"
    messagebox.showinfo("Prediction", f"The person will: {status}")


# Create the main window
window = tk.Tk()
window.title("Purchase Predictor")

# Create input fields and labels for Age and Salary
label_age = tk.Label(window, text="Age:")
label_age.grid(row=0, column=0, padx=10, pady=10)
entry_age = tk.Entry(window)
entry_age.grid(row=0, column=1, padx=10, pady=10)

label_salary = tk.Label(window, text="Salary:")
label_salary.grid(row=1, column=0, padx=10, pady=10)
entry_salary = tk.Entry(window)
entry_salary.grid(row=1, column=1, padx=10, pady=10)

# Create a button to predict purchase
predict_button = tk.Button(
    window, text="Predict Purchase", command=predict_purchase)
predict_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Run the GUI event loop
window.mainloop()
