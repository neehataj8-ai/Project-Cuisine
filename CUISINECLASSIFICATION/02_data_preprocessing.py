
from pathlib import Path
import pandas as pd

# Find and load the dataset
project_folder = Path(__file__).parent
csv_files = list(project_folder.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found. Put your dataset in the project folder."
    )

df = pd.read_csv(csv_files[0], low_memory=False)

print("Original dataset shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Check the target column
if "cuisines" not in df.columns:
    raise ValueError(
        "Cuisine column not found. Available columns: "
        + str(df.columns.tolist())
    )

# Remove rows without cuisine labels
df = df.dropna(subset=["cuisines"]).copy()

# Clean the cuisine text
df["cuisines"] = df["cuisines"].astype(str).str.strip()
df = df[df["cuisines"] != ""]

# Use the first cuisine as the target label
df["cuisine_label"] = df["cuisines"].str.split(",").str[0].str.strip()

# Save the cleaned dataset
output_path = project_folder / "cleaned_restaurants.csv"
df.to_csv(output_path, index=False)

print("\nData cleaning completed!")
print("Cleaned dataset shape:", df.shape)
print("\nTop 15 cuisine categories:")
print(df["cuisine_label"].value_counts().head(15))
print("\nSaved cleaned dataset to:", output_path.name)
