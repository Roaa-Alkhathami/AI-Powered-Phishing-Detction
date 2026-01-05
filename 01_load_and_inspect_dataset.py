import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# Step 1: Load the dataset
try:
    # Update the filename if needed
    df = pd.read_csv('dataset.csv')
    print(" Dataset loaded successfully.\n")
except FileNotFoundError:
    print(" CSV file not found. Make sure it's in the same directory.")
    exit()

# Step 2: Basic Info
print(" Dataset Shape:", df.shape)
print("\n First 5 Rows:\n", df.head())

print("\n Data Types & Non-Null Count:")
df.info()

# Step 3: Missing Value Check
print("\n Missing Values per Column:")
print(df.isnull().sum())

# Step 4: Class Distribution
if 'Type' in df.columns:
    print("\n🔐 Class Distribution (0 = Legitimate, 1 = Phishing):")
    print(df['Type'].value_counts())
    # Plot class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Type', data=df, palette="viridis")
    plt.title("Phishing vs Legitimate URLs")
    plt.xticks([0, 1], ['Legitimate', 'Phishing'])
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
else:
    print(" 'Type' column not found. Please check your dataset.")

# Step 5: Feature Summary
print("\n Statistical Summary (Numerical Features):")
print(df.describe())
