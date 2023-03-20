
from torch.utils.data import DataLoader
from torch import optim
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# The following training function has been adapted from it's original source found at: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#training-the-model
# The following modifications have been made
# - Progress indicators have been added
# - Validation data loader is accepted as a param to get validation scores
# - This function also takes a `SummaryWriter` object to write data to tensorboard
def train(epochs: int, data_loader: DataLoader, val_data_loader: DataLoader, model: nn.Module, criterion, optimizer: optim.Optimizer, writer: SummaryWriter, train_len: int, val_len: int):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)

  for epoch in tqdm(range(epochs), desc="Total Training Progress"):
    for phase in ['train', 'validate']:
      if phase == 'train':
        model.train()
      else:
        model.eval()

      running_loss = 0.0
      running_corrects = 0

      if phase == 'train':
        loader = data_loader
      else:
        loader = val_data_loader

      for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, predictions = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          if phase == 'train':
            loss.backward()
            optimizer.step()

        if phase == 'train':
          div = train_len
        else:
          div = val_len

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(predictions == labels.data)

        epoch_loss = running_loss / div
        epoch_acc = running_corrects.double() / div

      if phase == 'train':
        writer.add_scalar("train loss", epoch_loss, epoch)
        writer.add_scalar("train accuracy", epoch_acc, epoch)
        print(f"train loss {epoch_loss}")
        print(f"train accuracy {epoch_acc}")
      else:
        writer.add_scalar("validation loss", epoch_loss, epoch)
        writer.add_scalar("validation accuracy", epoch_acc, epoch)
        print(f"validation loss {epoch_loss}")
        print(f"validation accuracy {epoch_acc}")

  # Save model
  writer.close()

