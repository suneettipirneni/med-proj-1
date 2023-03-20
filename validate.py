from torch.utils.data import DataLoader
from torch import Tensor
from torch import nn

def validate(model: nn.Module, data_loader: DataLoader, criterion):
  model.eval()
  running_loss = 0.0
  for inputs, labels in data_loader:
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    running_loss += loss

  return running_loss / len(data_loader)