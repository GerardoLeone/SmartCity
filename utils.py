import copy
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

import config


def get_dataset(train_ratio=0.8):
    # Trasformazioni da applicare alle immagini
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Cambia la dimensione a 128x128
        transforms.ToTensor(),  # Converte l'immagine in tensore
        transforms.Normalize((0.5,), (0.5,))  # Normalizza i valori dei pixel
    ])

    # Carica il dataset
    dataset = datasets.ImageFolder(root='garbage_classification/', transform=transform)

    # Suddivisione in training e test set
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Suddivisione dei dati per utente
    user_group = {}
    indices = list(range(len(train_dataset)))
    split = len(indices) // config.NUM_USERS

    for i in range(config.NUM_USERS):
        start_idx = i * split
        end_idx = (i + 1) * split if i != config.NUM_USERS - 1 else len(indices)
        user_group[i] = indices[start_idx:end_idx]

    return train_dataset, test_dataset, user_group


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
