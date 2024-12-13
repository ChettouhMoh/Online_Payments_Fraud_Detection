import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
data = pd.read_csv("../data/PS_20174392719_1491204439457_log.csv")

print(data.head())

# Exploratory Data Analysis
print("Data shape:", data.shape)
print("Missing values:\n", data.isnull().sum())

# Transaction type distribution
type_counts = data['type'].value_counts()
fig = px.bar(x=type_counts.index, y=type_counts.values, labels={'x': 'Transaction Type', 'y': 'Count'}, title='Transaction Type Distribution')
fig.show()

# Preprocessing
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

# Splitting the data
X = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Train a Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
print("Model accuracy:", model.score(X_test, y_test))


# Example transaction
# new_transaction = [[1, 1000.0, 5000.0, 4000.0]]  # CASH_OUT, amount=1000, oldbalance=5000, newbalance=4000
# prediction = model.predict(new_transaction)
# print("Prediction:", prediction)  # Output will be 'No Fraud' or 'Fraud'

