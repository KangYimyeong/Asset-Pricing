from code.DataLoader import DataLoader
from code.utils import out_of_sample_R_square
from code.utils import gridSearch
from sklearn.ensemble import RandomForestRegressor


# Random Forest
def RF(train_data, val_data, test_data):
    # 数据集拆分
    test_X, test_y = DataLoader.data_split(test_data)
#     param_grid = {"n_estimators": [300], "max_depth": list(range(1, 7)), "max_features": [3, 5, 10, 20, 30, 50, 70, 100, 200, 500, 900]}
    param_grid = {"n_estimators": [300], "max_depth": list(range(1, 7)), "max_features": [3, 5, 10, 20, 30, 50, 100, 200], "n_jobs": [10]}
    # param_grid = {"n_estimators": [100], "max_features": [3, 5, 10], "n_jobs": [10]}
    res = gridSearch(model_builder=RandomForestRegressor, params=param_grid, train_data=train_data, val_data=val_data)
    best_model = res["best_estimator"]
#     prediction_plot(best_model.predict(test_X), test_y)
    print(f"RF R²: {out_of_sample_R_square(best_model.predict(test_X), test_y)}")
    return best_model.predict(test_X), test_y, best_model