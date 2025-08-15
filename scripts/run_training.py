from src.components.data_ingestion import CSVDataIngestion
from src.training.training import train
from src.evaluation.evaluate import evaluate

if __name__ == "__main__":
    CSVDataIngestion.run()
    train()
    evaluate()
