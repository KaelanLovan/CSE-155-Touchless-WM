import torch
import torch.nn as nn

class GestureGRU(nn.Module):
    def __init__(self, input_size=85, hidden_size=64, num_layers=1, num_classes=5):
        super(GestureGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # take output of last frame
        out = self.fc(out[:, -1, :])
        return out