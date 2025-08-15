import os
from dataclasses import dataclass
import torch

@dataclass
class Config:
    RAW_DATA_PATH = os.path.join('data', 'raw', 'data.csv')
    PROCESSED_TRAIN_PATH = os.path.join('data', 'processed', 'train.csv')
    PROCESSED_TEST_PATH = os.path.join('data', 'processed', 'test.csv')
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 20
    MODEL_SAVE_PATH = os.path.join('experiments', 'checkpoints', "model.pth")
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    INPUT_SHAPE = 178
    OUTPUT_SHAPE = 1
    HIDDEN_UNITS = 64
