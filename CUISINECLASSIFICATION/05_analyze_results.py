
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay
)

project_folder = Path(__file__).parent

# Load the saved training and testing data
X_train = pd.read_csv(project_folder / "X_train.csv")
X_test = pd.read_csv(project_folder / "X_test.csv")
y_train = pd.read_csv(project_folder / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(project_folder / "y_test.csv").iloc[:, 0]

# Identify feature types
numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

# Prepare numeric features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Prepare categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        max_categories=50
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Train Random Forest again for detailed analysis
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("Training Random Forest...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Save cuisine-wise performance report
report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(
    project_folder / "cuisine_performance.csv"
)

# Display the most and least accurately predicted cuisines
class_rows = report_df.loc[
    report_df.index.isin(y_test.unique())
].copy()

class_rows = class_rows.sort_values(
    "f1-score", ascending=False
)

print("\nTOP 10 CUISINES BY F1-SCORE")
print(class_rows.head(10).round(3).to_string())

print("\nBOTTOM 10 CUISINES BY F1-SCORE")
print(class_rows.tail(10).round(3).to_string())

# Save the actual class distribution
distribution = y_train.value_counts().rename_axis(
    "Cuisine"
).reset_index(name="Training_Count")

distribution.to_csv(
    project_folder / "cuisine_distribution.csv",
    index=False
)

# Plot the 15 most common cuisines in the training data
top_cuisines = distribution.head(15)

plt.figure(figsize=(12, 6))
plt.bar(
    top_cuisines["Cuisine"],
    top_cuisines["Training_Count"]
)
plt.xticks(rotation=60, ha="right")
plt.title("Top 15 Cuisines in Training Data")
plt.xlabel("Cuisine")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig(
    project_folder / "cuisine_distribution.png",
    dpi=150
)
plt.show()

# Plot a confusion matrix for the 10 most common test cuisines
top_test_cuisines = y_test.value_counts().head(10).index

mask = y_test.isin(top_test_cuisines)
actual_top = y_test[mask]
predicted_top = pd.Series(
    predictions, index=y_test.index
)[mask]

ConfusionMatrixDisplay.from_predictions(
    actual_top,
    predicted_top,
    labels=list(top_test_cuisines),
    xticks_rotation="vertical",
    cmap="Blues",
    colorbar=False
)

plt.title("Random Forest: Top 10 Test Cuisines")
plt.tight_layout()
plt.savefig(
    project_folder / "confusion_matrix.png",
    dpi=150
)
plt.show()

print("\nAnalysis completed!")
print("Saved cuisine_performance.csv")
print("Saved cuisine_distribution.csv")
print("Saved cuisine_distribution.png")
print("Saved confusion_matrix.png")
