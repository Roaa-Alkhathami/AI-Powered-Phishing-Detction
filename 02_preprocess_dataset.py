
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
df = pd.read_csv('dataset.csv')

# Step 2: Separate features (X) and target (y)
# X will contain all columns except 'Type'
# y will contain only the 'Type' column (0 = legitimate, 1 = phishing)
X = df.drop('Type', axis=1)
y = df['Type']

print("Separated features and target.")
print("X shape (features):", X.shape)
print("y shape (target):", y.shape)

# Step 3: Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature normalization completed.")

# Step 4: Split the data into training and testing sets (80% / 20%)
# stratify=y ensures the class distribution is preserved in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Train/Test split completed.")
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)
