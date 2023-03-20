import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# This code was taken from the following source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#test-the-network-on-the-test-data
# The only change applied to this code is the addition of a progress bar.
def test(model: nn.Module, data_loader: DataLoader):
  correct = 0
  total = 0

  with torch.no_grad():
    for inputs, labels in tqdm(data_loader, desc="Testing progress"):
      outputs = model(inputs)

      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print(f'Accuracy of the network on the {len(data_loader)} test images: {100 * correct // total} %')