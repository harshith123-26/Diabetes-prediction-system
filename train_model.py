import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ---------------- LOAD DATA ----------------
data = pd.read_csv("dataset/diabetes.csv")

# ---------------- SPLIT FEATURES & TARGET ----------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ---------------- ACCURACY ----------------
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nModel Training Completed Successfully ✅")
print("---------------------------------------")
print(f"Training Accuracy : {round(train_accuracy * 100, 2)}%")
print(f"Testing Accuracy  : {round(test_accuracy * 100, 2)}%")

# ---------------- CONFUSION MATRIX ----------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# ---------------- CLASSIFICATION REPORT ----------------
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and Scaler saved successfully as model.pkl and scaler.pkl")