
from pathlib import Path
import pandas as pd
import joblib

print("Starting cuisine prediction...")

project_folder = Path(__file__).parent
model_path = project_folder / "cuisine_model.joblib"

if not model_path.exists():
    raise FileNotFoundError("cuisine_model.joblib was not found!")

model = joblib.load(model_path)
print("Model loaded successfully!")

example = {
    "online_order": ["Yes"],
    "book_table": ["No"],
    "rate": [4.2],
    "votes": [120],
    "approx_cost(for two people)": [500],
    "rest_type": ["Casual Dining"],
    "listed_in(type)": ["Dine-out"],
    "listed_in(city)": ["Indiranagar"]
}

restaurant = pd.DataFrame(example)

expected_features = list(
    model.named_steps["preprocessor"].feature_names_in_
)

restaurant = restaurant.reindex(columns=expected_features)

prediction = model.predict(restaurant)[0]

print("\nPredicted cuisine:", prediction)
print("\nPrediction completed successfully!")
