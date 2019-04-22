import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class DQN(nn.Module): 
  def __init__(self, outputs=9):
    super(DQN, self).__init__()

    self.conv1 = nn.Conv2d(1, 8, kernel_size = 4, stride = 2)
    self.bn1 = nn.BatchNorm2d(8)
    self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 1, padding = 0)

    self.conv2 = nn.Conv2d(8, 16, kernel_size = 3)
    self.bn2 = nn.BatchNorm2d(16)
    self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 1, padding = 0)

    self.fc1 = nn.Linear(29584, 256)
    self.fc2 = nn.Linear(256, outputs)
  
  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = self.pool1(x)

    x = F.relu(self.bn2(self.conv2(x)))
    x = self.pool2(x)

    print(self.num_flat_features(x))
    # Flatten the input
    x = x.view(-1, 29584)
    x = F.relu(self.fc1(x))

    # Softmax activation for the last layer         

    return self.fc2(x)
  
  def num_flat_features(self, x):
    size = x.size()[1:]   # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features