
from pathlib import Path
import pandas as pd

# Find CSV files in the project folder
project_folder = Path(__file__).parent
csv_files = list(project_folder.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV file found in the project folder.")
else:
    print("CSV files found:")
    for i, file in enumerate(csv_files, start=1):
        print(f"{i}. {file.name}")

    # Load the first CSV file
    dataset_path = csv_files[0]
    df = pd.read_csv(dataset_path, low_memory=False)

    print("\nDataset loaded successfully!")
    print("File:", dataset_path.name)
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head().to_string())

    print("\nMissing values in each column:")
    print(df.isnull().sum().to_string())
