from src.components.data_ingestion import CSVDataIngestion
from src.training.training import train
from src.evaluation.evaluate import evaluate_model

if __name__ == "__main__":
    obj = CSVDataIngestion()
    obj.load_and_split_data()
    train()
    evaluate_model()
