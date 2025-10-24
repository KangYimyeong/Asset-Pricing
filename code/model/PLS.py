from code.DataLoader import DataLoader
from code.utils import out_of_sample_R_square
from code.utils import gridSearch
from sklearn.cross_decomposition import PLSRegression


# PLS (Partial Linear Square)
def PLS(train_data, val_data, test_data):
    test_X, test_y = DataLoader.data_split(test_data)
    param_grid = {"n_components": [1, 2, 3, 5, 10]}
    res = gridSearch(model_builder=PLSRegression, params=param_grid, train_data=train_data, val_data=val_data)
    best_model = res["best_estimator"]
    #     prediction_plot(best_model.predict(test_X), test_y)
    print(f"PLS R²: {out_of_sample_R_square(best_model.predict(test_X), test_y)}")
    return best_model.predict(test_X), test_y, best_model