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
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss, train_acc = 0, 0
        for batch, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
            y_pred_logits = model(X_batch)
            y_pred = torch.softmax(y_pred_logits, dim=1).argmax(dim=1)
            
            loss = loss_fn(y_pred_logits, y_batch)
            train_loss += loss.item()
            train_acc += accuracy_fn(y_true=y_batch, y_pred=y_pred)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        print(f"Epoch {epoch+1}")
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("Model saved.")

train()