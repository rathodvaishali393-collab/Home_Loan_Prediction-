import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "ml", "loan_dataset.csv")


FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area"
]


CATEGORICAL = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]


NUMERICAL = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]


def train_model():

    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURES].copy()
    y = df["Loan_Status"].astype(str).str.upper()

    preprocessor = ColumnTransformer([
        (
            "num",
            SimpleImputer(strategy="mean"),
            NUMERICAL
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            CATEGORICAL
        )
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    # Train model using complete dataset
    model.fit(X, y)

    return model


# Train model when Django starts
model = train_model()


def predict_loan(data):

    input_data = {}

    for feature in FEATURES:

        value = data.get(feature)

        if feature in NUMERICAL:

            if value in [None, ""]:
                input_data[feature] = None
            else:
                input_data[feature] = float(value)

        else:
            input_data[feature] = value

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    classes = list(model.named_steps["classifier"].classes_)

    probability_map = {
        str(cls): float(prob)
        for cls, prob in zip(classes, probabilities)
    }

    return {
        "prediction": str(prediction),
        "probability_yes": probability_map.get("Y", 0),
        "probability_no": probability_map.get("N", 0)
    }