import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from src.config.config import Config

class DataTransformation:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit_transform(self):
        """
        Fit the scaler to the data and transform it.
        """
        train_df = pd.read_csv(Config.PROCESSED_TRAIN_PATH)
        X = self.scaler.fit_transform(train_df.iloc[:, 1:-1])
        y = train_df.iloc[:, -1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)
    
    def transform_test(self):
        """
        Transform the test data using the fitted scaler.
        """
        test_df = pd.read_csv(Config.PROCESSED_TEST_PATH)
        X = self.scaler.transform(test_df.iloc[:, 1:-1])
        y = test_df.iloc[:, -1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)