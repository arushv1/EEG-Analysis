import torch
from src.config.config import Config
from src.components.data_transformation import DataTransformation
from src.components.datasets import EEGDataset
from torch.utils.data import DataLoader
from src.models.eeg_cnn import EEG_CNN
from src.utils import accuracy_fn
import torch.nn as nn

def evaluate_model():
    transformer = DataTransformation()
    X_test, y_test = transformer.transform_test()
    test_loader = DataLoader(EEGDataset(X_test.unsqueeze(1), y_test), batch_size=Config.BATCH_SIZE, shuffle=False)
    loss_fn = nn.BCEWithLogitsLoss()


    model = EEG_CNN(input_shape=Config.INPUT_SHAPE, hidden_units=Config.HIDDEN_UNITS, output_shape=Config.OUTPUT_SHAPE)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(Config.DEVICE)

    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
            # 1. Forward pass
            test_pred_logits = model(X_batch).squeeze()
            test_pred = torch.round(torch.sigmoid(test_pred_logits))
            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y_batch) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(y_true=y_batch, y_pred=test_pred)

        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_loader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_loader)

        ## Print out what's happening
        print(f"\n Test loss: {test_loss:.5f} Test acc: {test_acc:.2f}%\n")

