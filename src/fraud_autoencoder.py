
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# --------------------------
# Load Dataset
# --------------------------
print("Loading dataset...")

data = pd.read_csv("/workspaces/Fraud-detection-systems-/data/data/creditcard.csv")

print("Dataset shape:", data.shape)

# --------------------------
# Separate Features + Label
# --------------------------
X = data.drop("Class", axis=1)
y = data["Class"]

# --------------------------
# Normalize Data
# --------------------------
print("Scaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# Train only on NORMAL data
# --------------------------
X_train = X_scaled[y == 0]
X_test = X_scaled
y_test = y

print("Training samples:", X_train.shape)

# --------------------------
# Build AutoEncoder Model
# (lightweight for Codespaces)
# --------------------------
model = AutoEncoder(
    hidden_neuron_list=[32, 16, 16, 32],
    epoch_num=10,
    batch_size=512,
    contamination=0.001,
    verbose=1
)

# --------------------------
# Train
# --------------------------
print("Training model...")
model.fit(X_train)

# --------------------------
# Predict
# --------------------------
print("Detecting fraud...")

y_pred = model.predict(X_test)
scores = model.decision_function(X_test)

# --------------------------
# Evaluation
# --------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC AUC Score:", roc_auc_score(y_test, scores))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------
# Plot Reconstruction Error
# --------------------------
plt.figure(figsize=(8,5))

plt.hist(scores[y_test==0], bins=50, alpha=0.6, label="Normal", density=True)
plt.hist(scores[y_test==1], bins=50, alpha=0.6, label="Fraud", density=True)

plt.legend()
plt.title("Reconstruction Error Distribution")
plt.xlabel("Error Score")
plt.ylabel("Density")
plt.savefig("output.png")

