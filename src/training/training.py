import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.config.config import Config
from src.components.data_transformation import DataTransformation
from src.components.datasets import EEGDataset
from src.models.eeg_cnn import EEG_CNN
from src.utils import accuracy_fn

def train():
    transformer = DataTransformation()
    X_train, y_train = transformer.fit_transform()
    X_test, y_test = transformer.transform_test()

    train_loader = DataLoader(EEGDataset(X_train.unsqueeze(1), y_train), batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(EEGDataset(X_test.unsqueeze(1), y_test), batch_size=Config.BATCH_SIZE, shuffle=False)

    model = EEG_CNN(output_shape=Config.OUTPUT_SHAPE, input_shape=Config.INPUT_SHAPE, hidden_units=Config.HIDDEN_UNITS).to(Config.DEVICE)
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.EPOCHS):
        ## Training
        model.train()
        train_loss, train_acc = 0, 0
        for batch, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
            y_pred_logits = model(X_batch).squeeze()
            y_pred = torch.round(torch.sigmoid(y_pred_logits))
            
            loss = loss_fn(y_pred_logits, y_batch)
            train_loss += loss.item()
            train_acc += accuracy_fn(y_true=y_batch, y_pred=y_pred)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        ## Testing
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
        print(f"Epoch {epoch+1}")
        print(f"\nTrain loss: {train_loss:.5f} Train acc: {train_acc:.5f} | Test loss: {test_loss:.5f} Test acc: {test_acc:.2f}%\n")


    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("Model saved.")


if __name__ == "__main__":
    train()
    print("Training completed successfully.")