import torch
import torch.nn as nn
import torch.nn.functional as F

# Definizione della classe del modello CNNGarbage che estende nn.Module
class CNNGarbage(nn.Module):
    # Metodo di inizializzazione della classe
    def __init__(self, num_classes):
        super(CNNGarbage, self).__init__()
        # Primo strato di convoluzione: input con 3 canali, output con 32 canali, kernel di dimensione 3x3 e padding 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Secondo strato di convoluzione: input con 32 canali, output con 64 canali, kernel di dimensione 3x3 e padding 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Terzo strato di convoluzione: input con 64 canali, output con 128 canali, kernel di dimensione 3x3 e padding 1
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Quarto strato di convoluzione: input con 128 canali, output con 256 canali, kernel di dimensione 3x3 e padding 1
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # Strato di pooling massimo con finestra 2x2 e stride 2
        self.pool = nn.MaxPool2d(2, 2)
        # Strato di dropout con probabilità di disattivazione del 40%
        self.drop = nn.Dropout(0.4)
        # Primo strato fully connected: input di dimensione 256*8*8, output di dimensione 512
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        # Secondo strato fully connected: input di dimensione 512, output con numero di classi
        self.fc2 = nn.Linear(512, num_classes)
        # Strato di dropout per il fully connected layer con probabilità di disattivazione del 50%
        self.fc_drop = nn.Dropout(0.5)

        # Strati di batch normalization per ogni layer convoluzionale
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

    # Definizione del metodo forward per la propagazione in avanti del modello
    def forward(self, x):
        # Applicazione della prima convoluzione, batch norm, ReLU, max pooling e dropout
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(x)
        # Applicazione della seconda convoluzione, batch norm, ReLU, max pooling e dropout
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop(x)
        # Applicazione della terza convoluzione, batch norm, ReLU, max pooling e dropout
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop(x)
        # Applicazione della quarta convoluzione, batch norm, ReLU, max pooling e dropout
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.drop(x)
        # Rimodellamento del tensore per adattarlo al fully connected layer
        x = x.view(-1, 256 * 8 * 8)
        # Applicazione del primo fully connected layer e ReLU
        x = F.relu(self.fc1(x))
        # Applicazione del dropout al fully connected layer
        x = self.fc_drop(x)
        # Applicazione del secondo fully connected layer
        x = self.fc2(x)
        # Applicazione della funzione log softmax sull'output finale
        return F.log_softmax(x, dim=1)
