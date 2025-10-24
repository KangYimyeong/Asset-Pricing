from code.DataLoader import DataLoader
from code.utils import out_of_sample_R_square
from code.utils import gridSearch
from sklearn.ensemble import HistGradientBoostingRegressor


# GBRT + H
def GBRT(train_data, val_data, test_data):
    test_X, test_y = DataLoader.data_split(test_data)
    param_grid = {"max_depth": [1, 2], "learning_rate": [0.01, 0.1], "max_iter": [1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1000]}
    # param_grid = {"max_depth": [1, 2], "learning_rate": [0.01, 0.1], "max_iter": [1, 2, 5, 10, 20, 50, 100, 200, 400]}
    # param_grid = {"max_depth": [5, 10], "learning_rate": [0.1], "max_iter": [10, 50, 100]}
    res = gridSearch(model_builder=HistGradientBoostingRegressor, params=param_grid, train_data=train_data, val_data=val_data)
    best_model = res["best_estimator"]
#     prediction_plot(best_model.predict(test_X), test_y)
    print(f"GBRT R²: {out_of_sample_R_square(best_model.predict(test_X), test_y)}")
    return best_model.predict(test_X), test_y, best_model