import numpy as np

model_parameters = {
    "lightgbm": {
        "n_estimators": np.int64(610),
        "learning_rate": 0.009973363304630143,
        "num_leaves": np.int64(5),
        "max_depth": np.int64(19),
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "min_child_samples": np.int64(100),
        "subsample": 0.3,
        "colsample_bytree": 1.0,
    },
    "xgboost": {
        "n_estimators": np.int64(804),
        "learning_rate": 0.006125533332727532,
        "max_depth": np.int64(3),
        "subsample": 0.5572119099773152,
        "colsample_bytree": 0.8146634711869034,
        "gamma": 1.8038570285504327,
        "reg_alpha": 1.0,
        "reg_lambda": 0.5820494592800202,
    },
    "catboost": {
        "iterations": np.int64(609),
        "learning_rate": 0.01,
        "depth": np.int64(10),
        "l2_leaf_reg": 6.3321087265299525,
        "subsample": 0.5136845811222264,
    },
}
