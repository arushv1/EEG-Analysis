import torch.nn as nn
from src.config.config import Config

class EEG_CNN(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        # CNN layers for 1D signal
        self.conv_block = nn.Sequential(
            # Conv Block 1
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Conv Block 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Conv Block 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )
        
        # Fully connected layers
        self.fc_block = nn.Sequential(
            nn.Linear(128, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, output_shape)
        )
        
    def forward(self, x):
        # x shape: [batch_size, 1, sequence_length]
        x = self.conv_block(x)  # [batch_size, 128, 1]
        x = x.squeeze(-1)       # [batch_size, 128]
        x = self.fc_block(x)    # [batch_size, output_shape]
        return x
    
model_0 = EEG_CNN(
    input_shape=Config.INPUT_SHAPE,
    hidden_units=128,
    output_shape=Config.OUTPUT_SHAPE
)
model_0.to(Config.DEVICE)

print(model_0)