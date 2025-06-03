# gridsearch.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

RANDOM_STATE = 42


def read_train_data(data_path):
    X_train = pd.read_csv(os.path.join(data_path, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))

    return X_train, y_train

def train_model(best_params, X_train, y_train):
    # create model
    model = GradientBoostingRegressor(**best_params, random_state=RANDOM_STATE)
    model.fit(X_train, y_train.values.ravel())
    
    return model
    
    
def main():
    data_path = "data/processed"
    best_params_path = "models/best_params.pkl"
    
    # load bets parameters from grid search
    best_params = joblib.load(best_params_path)
    
    # load data
    X_train, y_train = read_train_data(data_path)
    
    # train model with best parameters
    model = train_model(best_params, X_train, y_train)
    
    # save model
    joblib.dump(model, 'models/gbr_model.pkl')
    

if __name__=="__main__":
    main()