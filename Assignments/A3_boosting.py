import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.datasets import load_breast_cancer

# Load dataset and convert to DataFrame
data = load_breast_cancer()
df = pd.DataFrame(data=np.column_stack((data.data, data.target)), columns=list(data.feature_names) + ["Target"])

# Separating features and target variable
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# Train-test split (Using stratify parameter for better class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initializing and fitting AdaBoost model using a lambda function
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
_ = (lambda model, X, y: model.fit(X, y))(ada_model, X_train, y_train)
ada_preds = ada_model.predict(X_test)
ada_accuracy = accuracy_score(y_test, ada_preds)

# Initializing and fitting XGBoost model with different method
xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=42)
xgb_model.set_params(eval_metric="logloss")  # Setting parameters separately
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_preds)

# Printing results
print(f"AdaBoost Accuracy: {ada_accuracy:.4f}")
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
