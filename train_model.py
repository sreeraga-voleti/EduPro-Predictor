import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("StudentAcademicData.csv")

# Convert Y/N columns to numeric
df["extra_curriculars"] = df["extra_curriculars"].map({"Y": 1, "N": 0})
df["part_time job"] = df["part_time job"].map({"Y": 1, "N": 0})

# Features for model
X = df[["Attendance", "Internal marks (out of 20)", "Age",
        "extra_curriculars", "part_time job", "backlogs"]]

# Target
y = df["Eligible for placement"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model & scaler
joblib.dump(model, "placement_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅ Model and Scaler saved successfully!")
