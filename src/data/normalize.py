from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os


def main():
    input_folderpath = "data/processed"
    output_folderpath = "data/processed"
    
    # Read Data
    X_test = pd.read_csv(os.path.join(input_folderpath, 'X_test.csv'))
    X_train = pd.read_csv(os.path.join(input_folderpath, 'X_train.csv'))
    
    # normalize data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # save data
    save_dataframes(X_train_scaled, X_test_scaled, output_folderpath)
    
    
def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    # Create folder if necessary
    os.makedirs(output_folderpath, exist_ok=True)
    
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)
        
    
if __name__ == '__main__':
    main()