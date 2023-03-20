from torchvision import models, datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomResizedCrop, CenterCrop, RandomHorizontalFlip, Resize, RandomGrayscale, RandomVerticalFlip
from torch.utils.data import DataLoader
from torch import optim
from os import path
from torch.nn import CrossEntropyLoss, Linear, Sequential, Dropout, ReLU, MaxPool2d, BatchNorm2d, BatchNorm1d
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from train import train
from test import test
from util import plot_results, class_accuracies

# This code is adapted/inspired from this source: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#finetuning-the-convnet
# It contains modifications specific to this assignment.

BATCH_SIZE = 2
NUM_EPOCHS = 30
INPUT_SIZE = 224

def run(task: str):
  writer = SummaryWriter()

  data_dir = "./data/chest_xray"

  transform = Compose([
    ToTensor(),
    Resize((INPUT_SIZE, INPUT_SIZE)),
  ])

  img_train = datasets.ImageFolder(path.join(data_dir, "train"), transform=Compose([
    ToTensor(),
    # Apply data augmentations to our input data
    RandomHorizontalFlip(),
    # Randomly crop to the desired size
    RandomResizedCrop((INPUT_SIZE, INPUT_SIZE)),
  ]))

  img_test = datasets.ImageFolder(path.join(data_dir, "test"), transform=transform)

  img_val = datasets.ImageFolder(path.join(data_dir, "val"), transform=Compose([
    ToTensor(),
    Resize((INPUT_SIZE, INPUT_SIZE)),
  ]))

  train_data_loader = DataLoader(img_train, batch_size=BATCH_SIZE, shuffle=True)
  test_data_loader = DataLoader(img_test, batch_size=BATCH_SIZE, shuffle=True)
  validation_data_loader = DataLoader(img_val, batch_size=BATCH_SIZE, shuffle=True)

  model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

  # Change the output size to the amount of classes in our dataset.
  model.fc = Sequential(
      Dropout(),
      Linear(model.fc.in_features, len(img_test.classes)),
  )

  criterion = CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters())

  if task == "train":
    train(model=model, epochs=NUM_EPOCHS, data_loader=train_data_loader, val_data_loader=validation_data_loader, criterion=criterion, optimizer=optimizer, writer=writer, train_len=len(img_train), val_len=len(img_val))
    torch.save(model, "./model-1.2.pth")

  if task == "test":
    model.load_state_dict(torch.load("./model-1.2.pth", map_location=torch.device('cpu')), strict=False)
    test(model=model, data_loader=test_data_loader)
    class_accuracies(model=model, test_loader=test_data_loader, classes=img_test.classes)