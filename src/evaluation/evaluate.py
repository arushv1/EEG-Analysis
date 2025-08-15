import torch
from src.config.config import Config
from src.components.data_transformation import DataTransformation
from src.components.datasets import EEGDataset
from torch.utils.data import DataLoader
from src.models.eeg_cnn import EEG_CNN

def evaluate():
    transformer = DataTransformation()
    X_test, y_test = transformer.transform_test()
    test_loader = DataLoader(EEGDataset(X_test.unsqueeze(1), y_test), batch_size=Config.BATCH_SIZE, shuffle=False)

    model = EEG_CNN(num_classes=len(y_test.unique()))
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.eval()

    correct, total = 0, 0
    with torch.inference():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")