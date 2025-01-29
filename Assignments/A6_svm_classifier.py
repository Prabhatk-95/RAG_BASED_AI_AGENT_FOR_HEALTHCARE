import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_decision_regions

# Load the Iris dataset from a CSV file
iris_df = pd.read_csv("iris.csv")

# Extract features and labels
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = iris_df[features].to_numpy()
y = iris_df["species"].to_numpy()

# Encode target labels
y = LabelEncoder().fit_transform(y)

# Handle missing values using median imputation
X = pd.DataFrame(X).fillna(X.mean()).to_numpy()

# Use only the first two features (sepal_length and sepal_width) for visualization
X_visual = X[:, :2]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_visual_train, X_visual_test = train_test_split(X_visual, test_size=0.3, random_state=42, stratify=y)

# Standardize feature data
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
X_visual_train, X_visual_test = scaler.transform(X_visual_train), scaler.transform(X_visual_test)

# Train SVM classifier on the full feature set
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42).fit(X_train, y_train)

# Predictions and evaluation
y_pred = svm_classifier.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris_df["species"].unique()))

# Train SVM classifier on 2D feature set for visualization
svm_visual_classifier = SVC(kernel='linear', C=1.0, random_state=42).fit(X_visual_train, y_train)

# Plot decision boundary using mlxtend
plt.figure(figsize=(8, 6))
plot_decision_regions(X_visual_train, y_train, clf=svm_visual_classifier, legend=2)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("SVM Decision Boundary (Training Set)")
plt.show()
