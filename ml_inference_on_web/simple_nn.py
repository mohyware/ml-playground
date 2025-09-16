import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(43)

# Let's say we have 10 input features, 16 neurons in the hidden layer, and 3 output classes
input_size = 10
hidden_size = 16
num_classes = 3

model = SimpleNet(input_size, hidden_size, num_classes)

fixed_values = [112, 212, 134, 4, 43, 3126, 7, 8321, 1, 10]
fixed_input_tensor = torch.tensor([fixed_values], dtype=torch.float32)

output = model(fixed_input_tensor)

print("Model Architecture:")
print(model)

print("\nOutput from the network:")
print(output)
