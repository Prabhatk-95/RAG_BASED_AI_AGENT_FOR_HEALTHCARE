import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset and create a DataFrame
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Separate features and target
X = df.iloc[:, :-1].values  # Extract feature columns
y = df['target'].values  # Extract target column

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Create and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
oob_score = rf_classifier.oob_score_

# Print results
print(f"Random Forest Classifier Accuracy: {accuracy:.2f}")
print(f"Out-of-Bag Score Estimate: {oob_score:.2f}")
