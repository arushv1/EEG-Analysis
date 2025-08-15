import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from src.config.config import Config
from src.exception import CustomException
from src.logger import logging
import sys
import joblib
import os

class DataTransformation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.scaler_path = "experiments/artifacts/scaler.pkl"
    
    def fit_transform(self):
        """
        Fit the scaler to the data and transform it.
        """
        try:
            train_df = pd.read_csv(Config.PROCESSED_TRAIN_PATH)
            
            X = train_df.iloc[:, 1:-1]
            X_scaled = self.scaler.fit_transform(X)
            y = train_df.iloc[:, -1]
            y = (train_df.iloc[:, -1] == 1).astype(int)
        
            # Save the fitted scaler
            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            joblib.dump(self.scaler, self.scaler_path)
            print(f"Scaler saved to {self.scaler_path}")
            
            logging.info("Training Data transformation completed successfully.")
            return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32)
    
        except Exception as e:
            raise CustomException(e, sys)
            
    
    def transform_test(self):
        """
        Transform the test data using the fitted scaler.
        """
        try:
            test_df = pd.read_csv(Config.PROCESSED_TEST_PATH)
            X = test_df.iloc[:, 1:-1]
            y = test_df.iloc[:, -1]
            y = (test_df.iloc[:, -1] == 1).astype(int)


            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(f"Scaler loaded from {self.scaler_path}")
            else:
                raise ValueError("Scaler not found. Run fit_transform() first!")
            
            X_scaled = self.scaler.transform(X)
            
            logging.info("Test Data transformation completed successfully.")
            return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32)
        
            
    
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    transformer = DataTransformation()
    X_train, y_train = transformer.fit_transform()
    X_test, y_test = transformer.transform_test()
    
    print("Transformed Training Data:", X_train.shape, y_train.shape)
    print("Transformed Test Data:", X_test.shape, y_test.shape)