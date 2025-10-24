from code.DataLoader import DataLoader
from code.utils import out_of_sample_R_square
from code.utils import gridSearch
from catboost import CatBoostRegressor


# CatBoost
def CatBoost(train_data, val_data, test_data):
    test_X, test_y = DataLoader.data_split(test_data)
    param_grid = {
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "iterations": [100, 200, 500, 1000],
        "l2_leaf_reg": [1, 3, 5, 7]
    }
    res = gridSearch(model_builder=CatBoostRegressor, params=param_grid, train_data=train_data, val_data=val_data)
    best_model = res["best_estimator"]
    print(f"CatBoost R²: {out_of_sample_R_square(best_model.predict(test_X), test_y)}")
    return best_model.predict(test_X), test_y, best_model
