import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
data = pd.read_csv("dataset/rainwater_dataset.csv")

# Convert text columns into numbers
le_roof = LabelEncoder()
le_location = LabelEncoder()

data["Roof_Type"] = le_roof.fit_transform(data["Roof_Type"])
data["Location"] = le_location.fit_transform(data["Location"])

# Features (Input)
X = data[["Roof_Area", "Rainfall", "Roof_Type", "Family_Size", "Location"]]

# Target (Output)
y = data["Harvested_Water"]

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/rainwater_model.pkl")

# Save encoders
joblib.dump(le_roof, "models/roof_encoder.pkl")
joblib.dump(le_location, "models/location_encoder.pkl")

print("✅ Model trained successfully!")
print("✅ Model saved in models/rainwater_model.pkl")