from torchvision import models, datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomResizedCrop, CenterCrop, Resize
from torch.utils.data import DataLoader
from torch import optim
from os import path
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter

from train import train
from test import test
from util import class_accuracies

# This code is adapted/inspired from this source: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#finetuning-the-convnet
# It contains modifications specific to this assignment.

BATCH_SIZE = 5
NUM_EPOCHS = 15
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
    Resize((INPUT_SIZE, INPUT_SIZE)),
  ]))

  img_test = datasets.ImageFolder(path.join(data_dir, "test"), transform=transform)

  img_val = datasets.ImageFolder(path.join(data_dir, "val"), transform=Compose([
    ToTensor(),
    Resize((INPUT_SIZE, INPUT_SIZE)),
  ]))

  train_data_loader = DataLoader(img_train, batch_size=BATCH_SIZE, shuffle=True)
  test_data_loader = DataLoader(img_test, batch_size=BATCH_SIZE, shuffle=True)
  validation_data_loader = DataLoader(img_val, batch_size=BATCH_SIZE, shuffle=True)

  num_classes = len(img_test.classes)
  model = models.resnet18(num_classes=num_classes, weights=None)

  # Try ADAM
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  criterion = CrossEntropyLoss()

  if task == "train":
    train(model=model, epochs=NUM_EPOCHS, data_loader=train_data_loader, val_data_loader=validation_data_loader, criterion=criterion, optimizer=optimizer, writer=writer, train_len=len(img_train), val_len=len(img_val))
    torch.save(model, "./model-1.1.pth")

  if task == "test":
    # Load the model
    model = models.resnet18(num_classes=2)
    model.load_state_dict(torch.load("./model-1.1.pth", map_location=torch.device('cpu')))
    test(model=model, data_loader=test_data_loader)
    class_accuracies(model=model, test_loader=test_data_loader, classes=img_test.classes)