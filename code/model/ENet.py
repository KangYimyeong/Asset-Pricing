import numpy as np
from code.DataLoader import DataLoader
from code.utils import out_of_sample_R_square
from code.utils import gridSearch
from sklearn.linear_model import ElasticNet


# Elastic Net + H
def ENet(train_data, val_data, test_data):
    test_X, test_y = DataLoader.data_split(test_data)

    param_grid = {"l1_ratio": [0.5], "alpha": np.logspace(-4, -1, 15)}
    res = gridSearch(model_builder=ElasticNet, params=param_grid, train_data=train_data, val_data=val_data)
    best_model = res["best_estimator"]
    #     prediction_plot(best_model.predict(test_X), test_y)
    print(f"ENet R²: {out_of_sample_R_square(best_model.predict(test_X), test_y)}")
    return best_model.predict(test_X), test_y, best_model