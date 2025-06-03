# evaluate.py

# gridsearch.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json

RANDOM_STATE = 42


def read_data(data_path):
    X_test = pd.read_csv(os.path.join(data_path, 'X_test_scaled.csv'))
    X_train = pd.read_csv(os.path.join(data_path, 'X_train_scaled.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))

    return X_train, X_test, y_train, y_test


def train_model(best_params, X_train, y_train):
    # create model
    model = GradientBoostingRegressor(**best_params, random_state=RANDOM_STATE)
    model.fit(X_train, y_train.values.ravel())
    
    return model
    
    
def main():
    data_path = "data/processed"
    model_path = "models/gbr_model.pkl"
    
    # load model
    model = joblib.load(model_path)
    
    # load data
    X_train, X_test, y_train, y_test = read_data(data_path)
    
    # predict
    y_pred = model.predict(X_test)
    
    # evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.4f}")
    print(f"R^2: {r2:.4f}")
    
    scores = {'MSE': mse,
              'R2': r2}
    
    # save metrics
    with open('metrics/scores.json', "w") as f:
        json.dump(scores, f, indent=4)
    
    # save predictions
    output_filepath = os.path.join(data_path, f'y_pred.csv')
    df_y_pred = pd.DataFrame(y_pred, columns=['predicted_silica_concentrate'])
    df_y_pred.to_csv(output_filepath, index=False)
    

if __name__=="__main__":
    main()