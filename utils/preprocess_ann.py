import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(filepath):

    df = pd.read_csv(filepath)

    # Remove unnecessary columns
    df = df.drop(
        ["RowNumber", "CustomerId", "Surname"],
        axis=1
    )

    # Convert categorical columns
    df = pd.get_dummies(
        df,
        columns=["Geography", "Gender"],
        drop_first=True
    )

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    print("Training Features:")
    print(X.columns.tolist())

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def preprocess_single_input(data):

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    return scaler.transform(data)