# make_dataset.py

import pandas as pd
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
import os


RANDOM_STATE = 42


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    input_filepath = "data/raw/raw.csv"
    output_folderpath = "data/processed"

    process_data(input_filepath, output_folderpath)


def process_data(input_filepath, output_folderpath):
    # Import datasets
    df = pd.read_csv(input_filepath)

    # Drop unnecessary columns
    df = df.drop(['date'], axis=1)

    # no further preprocessing because data is assured to have:
    # - dtypes are all float64
    # - no missing data
    # - no duplicates
    
    # Extract target and features
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=RANDOM_STATE)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train, X_test, y_train, y_test, output_folderpath)
    

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Create folder if necessary
    os.makedirs(output_folderpath, exist_ok=True)
    
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()