import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ensure the models directory exists
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)  # ✅ Creates 'models/' if it doesn't exist

# Load dataset
data = pd.read_csv("large_crm.csv")

# Select features and target
X = data.drop(columns=['CustomerID', 'Name', 'Email', 'Phone', 'Target'])  # Drop non-relevant columns
y = data['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(clf, os.path.join(model_dir, "random_forest.pkl"))  # ✅ Ensures path exists

# Evaluate model
accuracy = accuracy_score(y_test, clf.predict(X_test))
print(f"✅ Model trained and saved successfully! Accuracy: {accuracy:.2f}")
