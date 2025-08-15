import torch
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.config.config import Config
from src.exception import CustomException
from src.logger import logging
import sys

class CSVDataIngestion:
    def __init__(self):
        self.ingestion_config = Config()
    
    def load_and_split_data(self):
        logging.info("Entered the data ingestion method or component")
        try:
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
        except Exception as e:
            raise CustomException(e, sys)
        #return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path


if __name__ == "__main__":
    obj = CSVDataIngestion()
    obj.load_and_split_data()


