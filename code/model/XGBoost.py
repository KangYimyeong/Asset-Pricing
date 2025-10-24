from code.DataLoader import DataLoader
from code.utils import out_of_sample_R_square
from code.utils import gridSearch
from xgboost import XGBRegressor


# XGBoost
def XGBoost(train_data, val_data, test_data):
    test_X, test_y = DataLoader.data_split(test_data)
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 500, 1000],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0]
    }
    res = gridSearch(model_builder=XGBRegressor, params=param_grid, train_data=train_data, val_data=val_data)
    best_model = res["best_estimator"]
    print(f"XGBoost R²: {out_of_sample_R_square(best_model.predict(test_X), test_y)}")
    return best_model.predict(test_X), test_y, best_model
