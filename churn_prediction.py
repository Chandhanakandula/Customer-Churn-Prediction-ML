# Customer Churn Prediction using Machine Learning

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
data = pd.read_csv("churn.csv")

print("Dataset Loaded Successfully")
print(data.head())

# -----------------------------
# STEP 2: Data Cleaning
# -----------------------------

# Drop customerID (not useful)
data.drop("customerID", axis=1, inplace=True)

# Replace blank spaces with NaN
data.replace(" ", np.nan, inplace=True)

# Drop missing values
data.dropna(inplace=True)

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"])

# -----------------------------
# STEP 3: Encode Categorical Data
# -----------------------------
le = LabelEncoder()

for column in data.columns:
    if data[column].dtype == "object":
        data[column] = le.fit_transform(data[column])

# -----------------------------
# STEP 4: Split Features & Target
# -----------------------------
X = data.drop("Churn", axis=1)
y = data["Churn"]

# -----------------------------
# STEP 5: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 6: Logistic Regression Model
# -----------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, lr_pred))

# -----------------------------
# STEP 7: Random Forest Model
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# -----------------------------
# STEP 8: Prediction on New Customer
# -----------------------------
sample_customer = X.iloc[[0]]
result = rf.predict(sample_customer)
if result[0] == 1:
    print("\nPrediction: Customer will CHURN")
else:
    print("\nPrediction: Customer will NOT churn")
