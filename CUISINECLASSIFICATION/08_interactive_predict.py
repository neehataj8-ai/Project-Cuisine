
from pathlib import Path

import joblib
import pandas as pd

PROJECT_FOLDER = Path(__file__).parent
MODEL_PATH = PROJECT_FOLDER / "cuisine_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "cuisine_model.joblib is missing. Run 06_save_model.py first."
    )

model = joblib.load(MODEL_PATH)
expected_features = list(
    model.named_steps["preprocessor"].feature_names_in_
)


def ask_text(prompt):
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Please enter a value.")


def ask_number(prompt, minimum=0, maximum=None):
    while True:
        try:
            value = float(input(prompt))
            if value < minimum or (
                maximum is not None and value > maximum
            ):
                print(f"Enter a number from {minimum} to {maximum}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


print("\n================================")
print(" RESTAURANT CUISINE CLASSIFIER")
print("================================")
print("Enter restaurant details to predict its cuisine.\n")

while True:
    restaurant = {
        "online_order": [ask_text("Online order (Yes/No): ")],
        "book_table": [ask_text("Table booking (Yes/No): ")],
        "rate": [ask_number("Rating (0-5): ", 0, 5)],
        "votes": [ask_number("Number of votes: ")],
        "approx_cost(for two people)": [
            ask_number("Approximate cost for two people (Rs): ")
        ],
        "rest_type": [ask_text("Restaurant type: ")],
        "listed_in(type)": [ask_text("Service type (e.g. Dine-out): ")],
        "listed_in(city)": [ask_text("Area/city (e.g. Indiranagar): ")],
    }

    restaurant_df = pd.DataFrame(restaurant)
    restaurant_df = restaurant_df.reindex(columns=expected_features)

    prediction = model.predict(restaurant_df)[0]

    print("\n--------------------------------")
    print(f"Predicted cuisine: {prediction}")
    print("This is a prediction, not a guarantee.")
    print("--------------------------------")

    again = input(
        "\nPredict another restaurant? (yes/no): "
    ).strip().lower()

    if again not in ("yes", "y"):
        print("\nThank you! Program finished.")
        break

