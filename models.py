import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNGarbage(nn.Module):
    def __init__(self, num_classes):
        super(CNNGarbage, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.4)  # Aumentato il dropout a 0.4
        self.fc1 = nn.Linear(256 * 8 * 8, 512)  # Cambiato il dimensionamento a 256
        self.fc2 = nn.Linear(512, num_classes)
        self.fc_drop = nn.Dropout(0.5)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(x)  # Aggiunto il dropout dopo il pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop(x)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.drop(x)
        x = x.view(-1, 256 * 8 * 8)  # Aggiornato per la nuova dimensione
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
