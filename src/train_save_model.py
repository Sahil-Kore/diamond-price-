import model_dispatcher
import optimal_hyperparameters

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import root_mean_squared_error

import argparse


def save_predictions_and_model(model, features, name, version):
    test_df = pd.read_csv("../inputs/test_lb.csv")
    sub_df = pd.read_csv("../inputs/submission.csv")

    X_test = test_df[features]

    y_preds = model.predict(X_test)
    sub_df["price"] = y_preds

    sub_df.to_csv(f"../outputs/{name}{version}.csv", index=False)
    joblib.dump(model, f"../models/{name}{version}.joblib")


def train_model(params, model_class, features, df, name, version):
    model = model_class(**params)

    X = df[features]
    y = df["price"]

    model.fit(X, y)
    rmse = root_mean_squared_error(y, model.predict(X))
    print(f" error is :{rmse}")

    save_predictions_and_model(model, features, name, version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_class",
        type=str,
    )
    parser.add_argument(
        "--version",
        type=str,
    )

    args = parser.parse_args()
    version = args.version

    model_bundle = model_dispatcher.models[args.model_class]
    model_class = model_bundle["model_class"]

    params = optimal_hyperparameters.model_parameters[args.model_class]

    train_df = pd.read_csv("../inputs/train_lb.csv")
    features = [c for c in train_df.columns if c not in "price"]
    train_model(
        params, model_class, features, train_df, args.model_class, f"_tuned_{version}"
    )
