import pandas as pd
import numpy as np

import cudf
import cupy as cp


from functools import partial

from skopt import gp_minimize
from skopt.plots import plot_convergence

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

import argparse

import model_dispatcher

import matplotlib.pyplot as plt

import warnings


def optimize(params, param_names, model_class, df, features, fixed_params):
    params = dict(zip(param_names, params))
    kwargs = params.copy()
    kwargs = kwargs | fixed_params

    model = model_class(**kwargs)

    kf = KFold(n_splits=5)

    rmse = []

    for train_dx, val_idx in kf.split(df):
        train_df = df.iloc[train_dx]
        val_df = df.iloc[val_idx]

        X_train = train_df[features]
        X_val = val_df[features]

        y_train = train_df["price"]
        y_val = val_df["price"]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        if "cudf" in str(type(y_val)):
            y_true_np = y_val.to_numpy()  # Use .to_numpy() or .get()
        else:  # It's a pandas Series
            y_true_np = y_val.values

        if "cupy" in str(type(preds)):
            preds_np = preds.get()  # Convert from cupy array to numpy array
        elif "cudf" in str(type(preds)):
            preds_np = preds.to_numpy()
        else:  # It's already a numpy array from sklearn/lgbm
            preds_np = preds
        fold_rmse = root_mean_squared_error(y_true_np, preds_np)
        rmse.append(fold_rmse)
    print(np.mean(rmse))
    return np.mean(rmse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_class",
        type=str,
    )
    args = parser.parse_args()
    model_bundle = model_dispatcher.models[args.model_class]

    param_space = model_bundle["param_space"]
    param_names = model_bundle["param_names"]
    model_class = model_bundle["model_class"]
    fixed_params = model_bundle["fixed_params"]
    uses_gpu = model_bundle["uses_gpu"]

    if uses_gpu:
        df = cudf.read_csv("../inputs/train_lb.csv")
        # df = df.sample(n = 5000).reset_index(drop = True)
    else:
        df = pd.read_csv("../inputs/train_lb.csv")
        # df = df.sample(n = 5000).reset_index(drop = True)

    features = df.drop(["id", "price"], axis=1).columns
    optimization_function = partial(
        optimize,
        param_names=param_names,
        model_class=model_class,
        df=df,
        features=features,
        fixed_params=fixed_params,
    )

    print("Starting hyperparameter optimization")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = gp_minimize(
            optimization_function,
            dimensions=param_space,
            n_calls=50,
            n_random_starts=5,
        )
    best_params = dict(zip(param_names, result["x"]))
    best_rmse = result.fun
    print("Model class ", args.model_class)
    print(f"best_params are {best_params}")
    print(f"Lowest rmse is {best_rmse}")
    plot_convergence(result)
    plt.show()

# lightgbm {'n_estimators': np.int64(610), 'learning_rate': 0.009973363304630143, 'num_leaves': np.int64(5), 'max_depth': np.int64(19), 'reg_alpha': 0.0, 'reg_lambda': 1.0, 'min_child_samples': np.int64(100), 'subsample': 0.3, 'colsample_bytree': 1.0}
# xgboost{'n_estimators': np.int64(804), 'learning_rate': 0.006125533332727532, 'max_depth': np.int64(3), 'subsample': 0.5572119099773152, 'colsample_bytree': 0.8146634711869034, 'gamma': 1.8038570285504327, 'reg_alpha': 1.0, 'reg_lambda': 0.5820494592800202}
# catboost {'iterations': np.int64(609), 'learning_rate': 0.01, 'depth': np.int64(10), 'l2_leaf_reg': 6.3321087265299525, 'subsample': 0.5136845811222264}
