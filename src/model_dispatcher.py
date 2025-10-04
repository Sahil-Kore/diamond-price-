import lightgbm as lgb
from skopt import space
from cuml.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb
from cuml.linear_model import LinearRegression
from cuml.linear_model import Lasso
from cuml.linear_model import Ridge
from cuml.svm import SVR
from catboost import CatBoostRegressor
import cupy as cp

import pandas as pd

df = pd.read_csv("../inputs/train.csv")
df
models = {
    "features": {
        "carat",
        "cut",
        "color",
        "clarity",
        "depth",
        "table",
        "x",
        "y",
        "z",
        "price",
    },
    "lightgbm": {
        "model_class": lgb.LGBMRegressor,
        "param_names": [
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "reg_alpha",
            "reg_lambda",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
        ],
        "param_space": [
            space.Integer(low=10, high=1000, name="n_estimators"),
            space.Real(low=0.001, high=0.01, prior="uniform", name="learning_rate"),
            space.Integer(low=5, high=50, name="num_leaves"),
            space.Integer(low=1, high=20, name="max_depth"),
            space.Real(low=0.0, high=1.0, prior="uniform", name="reg_alpha"),
            space.Real(low=0, high=1.0, prior="uniform", name="reg_lambda"),
            space.Integer(low=1, high=100, name="min_child_samples"),
            space.Real(low=0.3, high=1.0, prior="uniform", name="subsample"),
            space.Real(low=0.3, high=1.0, prior="uniform", name="colsample_bytree"),
        ],
        "fixed_params": {"device": "gpu", "verbose": -1},
        "uses_gpu": False,
    },
    "xgboost": {
        "model_class": xgb.XGBRegressor,
        "param_names": [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        ],
        "param_space": [
            space.Integer(low=10, high=1000, name="n_estimators"),
            space.Real(0.001, 0.01, prior="uniform", name="learning_rate"),
            space.Integer(1, 12, name="max_depth"),
            space.Real(0.3, 1.0, prior="uniform", name="subsample"),
            space.Real(0.3, 1.0, prior="uniform", name="colsample_bytree"),
            space.Real(0.0, 5.0, prior="uniform", name="gamma"),
            space.Real(0.0, 1.0, prior="uniform", name="reg_alpha"),
            space.Real(0.0, 1.0, prior="uniform", name="reg_lambda"),
        ],
        "fixed_params": {
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "device": "gpu",
            "eval_metric": "rmse",
        },
        "features": [
            "RhythmScore",
            "AudioLoudness",
            "VocalContent",
            "AcousticQuality",
            "InstrumentalScore",
            "LivePerformanceLikelihood",
            "MoodScore",
            "TrackDurationMs",
        ],
        "uses_gpu": True,
    },
    "random_forest": {
        "model_class": RandomForestRegressor,
        "param_names": [
            "n_estimators",
            "max_depth",
            "max_features",
            "min_samples_split",
            "min_samples_leaf",
        ],
        "param_space": [
            space.Integer(100, 1000, name="n_estimators"),
            space.Integer(3, 20, name="max_depth"),
            space.Real(0.1, 1.0, prior="uniform", name="max_features"),
            space.Integer(2, 20, name="min_samples_split"),
            space.Integer(1, 20, name="min_samples_leaf"),
        ],
        "fixed_params": {},
    },
    "hist_gradient_boosting": {
        "model_class": HistGradientBoostingRegressor,
        "param_names": [
            "max_iter",
            "learning_rate",
            "max_leaf_nodes",
            "max_depth",
            "min_samples_leaf",
            "l2_regularization",
        ],
        "param_space": [
            space.Integer(100, 1000, name="max_iter"),
            space.Real(0.01, 0.1, prior="uniform", name="learning_rate"),
            space.Integer(10, 150, name="max_leaf_nodes"),
            space.Integer(3, 20, name="max_depth"),
            space.Integer(2, 100, name="min_samples_leaf"),
            space.Real(0.0, 1.0, prior="uniform", name="l2_regularization"),
        ],
        "fixed_params": {"verbose": 0},
        "uses_gpu": False,
    },
    "linear_regression": {
        "model_class": LinearRegression,
        "param_names": ["fit_intercept", "algorithm"],
        "param_space": [
            space.Categorical([True, False], name="fit_intercept"),
            space.Categorical(
                ["svd", "eig", "qr", "svd-qr", "svd-jacobi"], name="algorithm"
            ),
        ],
        "fixed_params": {},
        "uses_gpu": True,
    },
    "lasso": {
        "model_class": Lasso,
        "param_names": ["alpha", "fit_intercept"],
        "param_space": [
            # Regularization strength is best searched on a log scale
            space.Real(1e-4, 1e2, prior="log-uniform", name="alpha"),
            space.Categorical([True, False], name="fit_intercept"),
        ],
        "fixed_params": {},
        "uses_gpu": True,
    },
    "ridge": {
        "model_class": Ridge,
        "param_names": ["alpha", "fit_intercept"],
        "param_space": [
            space.Real(1e-4, 1e2, prior="log-uniform", name="alpha"),
            space.Categorical([True, False], name="fit_intercept"),
        ],
        "fixed_params": {},
        "uses_gpu": True,
    },
    "svm": {
        "model_class": SVR,  # SVR is the regressor version of SVM
        "param_names": ["C", "gamma", "kernel"],
        "param_space": [
            # C and gamma are also best searched on a log scale
            space.Real(1e-2, 1e3, prior="log-uniform", name="C"),
            space.Real(1e-4, 1e1, prior="log-uniform", name="gamma"),
            space.Categorical(["rbf", "linear", "poly", "sigmoid"], name="kernel"),
        ],
        "fixed_params": {},
        "uses_gpu": True,
    },
    "catboost": {
        "model_class": CatBoostRegressor,
        "param_names": [
            "iterations",
            "learning_rate",
            "depth",
            "l2_leaf_reg",
            "subsample",
        ],
        "param_space": [
            space.Integer(low=500, high=800, name="iterations"),
            space.Real(0.001, 0.01, prior="uniform", name="learning_rate"),
            space.Integer(3, 10, name="depth"),
            space.Real(1.0, 10.0, prior="uniform", name="l2_leaf_reg"),
            space.Real(0.5, 1.0, prior="uniform", name="subsample"),
        ],
        "fixed_params": {
            "verbose": False,
            "task_type": "GPU",
            "bootstrap_type": "Poisson",
        },
        "uses_gpu": False,
    },
}
