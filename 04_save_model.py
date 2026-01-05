# 06_save_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Load dataset
df = pd.read_csv('dataset.csv')

# Step 2: Split features and target
X = df.drop('Type', axis=1)
y = df['Type']

# Step 3: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model training completed.")

# Step 6: Save the trained model using pickle
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Step 7: Save the scaler too (important for preprocessing new inputs)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully as random_forest_model.pkl and scaler.pkl")
