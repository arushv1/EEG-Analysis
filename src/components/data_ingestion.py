import torch
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.config.config import Config

class CSVDataIngestion:
    def __init__(self):
        self.ingestion_config = Config()
    
    def load_and_split_data(self):
        #Read the CSV file
        data_df = pd.read_csv(Config.RAW_DATA_PATH)
        
        #Ensure output dir exists, if not create it
        os.makedirs(os.path.dirname(Config.PROCESSED_TRAIN_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(Config.PROCESSED_TEST_PATH), exist_ok=True)

        #Save the raw data to the specified path
        data_df.to_csv(self.ingestion_config.RAW_DATA_PATH, index=False)

        train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)

        train_df.to_csv(self.ingestion_config.PROCESSED_TRAIN_PATH, index=False)
        test_df.to_csv(self.ingestion_config.PROCESSED_TEST_PATH, index=False)
        print("Data ingestion completed successfully.")
        #return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path


if __name__ == "__main__":
    obj = CSVDataIngestion()
    obj.load_and_split_data()


'''
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float32)
        label = torch.tensor(self.data.iloc[idx, -1], dtype=torch.long)
        return features, label
    

if __name__ == "__main__":
    #Step 1: Ingest Data
    ingestion = CSVDataIngestion()
    train_path, test_path = ingestion.load_and_split_data('notebook/data/data.csv')

    #Step 2: Create datasets
    train_dataset = CustomDataset(train_path)
    test_dataset = CustomDataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for X, y in train_loader:
        print(X.shape, y.shape)
        break







obj = DataIngestion()
obj.initiate_data_ingestion()

'''