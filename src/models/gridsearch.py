# gridsearch.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import yaml

RANDOM_STATE = 42


def read_train_data(data_path):
    # X_test = pd.read_csv(os.path.join(data_path, 'X_test_scaled.csv'))
    X_train = pd.read_csv(os.path.join(data_path, 'X_train_scaled.csv'))
    # y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))

    return X_train, y_train


def run_grid_search(param_grid, X_train, y_train):
    # create model
    model = GradientBoostingRegressor(random_state=RANDOM_STATE)

    # init grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    # run grid search for best parameters
    grid_search.fit(X_train, y_train.values.ravel())
    
    return grid_search
    
def main():
    data_path = "data/processed"
    
    X_train, y_train = read_train_data(data_path)
    
    # read parameter space
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    param_grid = config["gridsearch"]
    
    grid_search = run_grid_search(param_grid, X_train, y_train)
    
    print("Best params:", grid_search.best_params_)
    
    # save best parameters to file
    joblib.dump(grid_search.best_params_, 'models/best_params.pkl')
    

if __name__=="__main__":
    main()