import torch
from config import *

class LinearClassification(torch.nn.Module):
    def __init__(self, num_classes):
        super(LinearClassification, self).__init__()
        self.fc1 = torch.nn.Linear(INPUT_HIDDEN_SIZE, HIDDEN_SIZE)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(HIDDEN_SIZE, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Apply the linear layer and ReLU to each time step
        x = self.fc1(x)  # x shape (B, L, H) -> (B, L, hidden_size)
        x = self.relu(x) 
        
        # Average over the time steps (L dimension)
        x = x.mean(dim=1)  # Now x has shape (B, hidden_size)
        
        x = self.fc2(x)    # Now applying the final layer (B, hidden_size) -> (B, num_classes)
        x = self.softmax(x)
        return x