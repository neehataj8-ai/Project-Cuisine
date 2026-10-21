
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
project_folder = Path(__file__).parent
df = pd.read_csv(project_folder / "cleaned_restaurants.csv")

print("Dataset loaded:", df.shape)

# Choose useful features that exist in this dataset
possible_features = [
    "online_order",
    "book_table",
    "rate",
    "votes",
    "approx_cost(for two people)",
    "rest_type",
    "listed_in(type)",
    "listed_in(city)"
]

features = [col for col in possible_features if col in df.columns]

# Our target is the cuisine category
X = df[features].copy()
y = df["cuisine_label"].copy()

print("\nFeatures selected:")
print(features)

# Convert rating and cost to numeric values
if "rate" in X.columns:
    X["rate"] = (
        X["rate"].astype(str)
        .str.replace("/5", "", regex=False)
        .str.strip()
        .replace({"NEW": None, "-": None})
    )
    X["rate"] = pd.to_numeric(X["rate"], errors="coerce")

if "approx_cost(for two people)" in X.columns:
    X["approx_cost(for two people)"] = (
        X["approx_cost(for two people)"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    X["approx_cost(for two people)"] = pd.to_numeric(
        X["approx_cost(for two people)"],
        errors="coerce"
    )

if "votes" in X.columns:
    X["votes"] = pd.to_numeric(X["votes"], errors="coerce")

# Drop columns that contain no usable values
X = X.dropna(axis=1, how="all")

# Remove cuisine classes with fewer than 2 records
counts = y.value_counts()
keep = y.isin(counts[counts >= 2].index)
X = X.loc[keep].copy()
y = y.loc[keep].copy()

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save the split data for the next stage
X_train.to_csv(project_folder / "X_train.csv", index=False)
X_test.to_csv(project_folder / "X_test.csv", index=False)
y_train.to_csv(project_folder / "y_train.csv", index=False)
y_test.to_csv(project_folder / "y_test.csv", index=False)

print("\nData preparation completed!")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Number of cuisine categories:", y.nunique())
print("Training and testing CSV files saved.")
