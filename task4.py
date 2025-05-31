
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load and preprocess dataset
df = pd.read_csv(r'C:\Users\hp\Downloads\data.csv') 

# Drop unnecessary columns
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# Encode target column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Feature and target split
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Train/test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Train logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 3: Predictions and evaluation
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Step 4: Threshold tuning
custom_threshold = 0.3
y_pred_custom = (y_prob >= custom_threshold).astype(int)
conf_matrix_custom = confusion_matrix(y_test, y_pred_custom)
report_custom = classification_report(y_test, y_pred_custom)

# Step 5: Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print("=== Logistic Regression Evaluation ===")
print("\n[1] Confusion Matrix (Default Threshold 0.5):")
print(conf_matrix)

print("\n[2] Classification Report (Default Threshold 0.5):")
print(report)

print(f"[3] ROC-AUC Score: {auc:.4f}")

print("\n[4] Confusion Matrix (Custom Threshold = 0.3):")
print(conf_matrix_custom)

print("\n[5] Classification Report (Custom Threshold = 0.3):")
print(report_custom)

print("\n[6] Sigmoid Function Overview:")
print("The sigmoid function maps any real value into a range between 0 and 1, and is used to convert the linear output of logistic regression into a probability.")

# === Plot ROC Curve and Sigmoid ===
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(12, 5))

# Plot 1: ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()

# Plot 2: Sigmoid Function
z = np.linspace(-10, 10, 200)
sig = sigmoid(z)
plt.subplot(1, 2, 2)
plt.plot(z, sig)
plt.axhline(y=custom_threshold, color='r', linestyle='--', label=f'Threshold = {custom_threshold}')
plt.title("Sigmoid Function")
plt.xlabel("z (Logit Output)")
plt.ylabel("Sigmoid(z)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()



