
from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Project folder
project_folder = Path(__file__).parent

# Load training and testing data
X_train = pd.read_csv(project_folder / "X_train.csv")
X_test = pd.read_csv(project_folder / "X_test.csv")

y_train = pd.read_csv(project_folder / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(project_folder / "y_test.csv").iloc[:, 0]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Identify numeric and categorical columns
numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

# Preprocess numerical data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocess text/category data
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        max_categories=50
    ))
])

# Apply the correct preprocessing to each column
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Create two complete machine learning pipelines
models = {
    "Logistic Regression": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]),

    "Random Forest": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# Train and evaluate both models
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions,
        average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, predictions,
        average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions,
        average="weighted", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    if name == "Random Forest":
        print("\nRandom Forest classification report:")
        print(classification_report(
            y_test, predictions, zero_division=0
        ))

# Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv(
    project_folder / "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print("\nSaved results to model_comparison.csv")
