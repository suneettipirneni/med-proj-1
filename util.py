import matplotlib.pyplot as plt
import json
from os import path
from torch import nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

name_mappings = {
  'val-loss': "Validation Loss",
  'train-loss': "Train Loss"
}

def plot_results(results_dir: str, title: str):

  for result_type in ['val-loss', 'train-loss']:
    loss_file = open(path.join(results_dir, f'{result_type}.json'))
    results = json.load(loss_file)

    x = []
    y = []
    
    for result in results:
      _, epoch, value = result
      x.append(epoch)
      y.append(value)
    
    plt.plot(x, y, label=name_mappings[result_type])
    

  plt.title(title)
  plt.legend(loc="upper left")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.show()

# This code is taken from training section of the pytorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network
# It has been adapted to include a progress bar.
def class_accuracies(model: nn.Module, test_loader: DataLoader, classes: list[str]):
  # prepare to count predictions for each class
  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}

  # again no gradients needed
  with torch.no_grad():
      for data in tqdm(test_loader):
          images, labels = data
          outputs = model(images)
          _, predictions = torch.max(outputs, 1)
          # collect the correct predictions for each class
          for label, prediction in zip(labels, predictions):
              if label == prediction:
                  correct_pred[classes[label]] += 1
              total_pred[classes[label]] += 1


  # print accuracy for each class
  for classname, correct_count in correct_pred.items():
      accuracy = 100 * float(correct_count) / total_pred[classname]
      print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# plot_results('./results/task1.1', 'Model 1.1')