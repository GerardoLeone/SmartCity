import copy
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

import config


def get_dataset(train_ratio=0.8):
    # Trasformazioni da applicare alle immagini
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(30),  # Aumenta la rotazione casuale fino a 30 gradi
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # Aggiungi flip verticale casuale
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Aggiungi variazioni di colore
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x))
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


def average_weights(w, sample_counts):
    """
    Averages the weights using sample counts as weights.
    Args:
      w: A list of state_dicts from different models.
      sample_counts: A list of integers representing the number of samples for each client.
    Returns:
      The averaged state_dict.
    """
    total_samples = sum(sample_counts)
    averaged_weights = copy.deepcopy(w[0])
    for key in averaged_weights.keys():
        for i in range(len(w)):
            if i == 0:
                averaged_weights[key] = w[i][key] * (sample_counts[i] / total_samples)
            else:
                averaged_weights[key] += w[i][key] * (sample_counts[i] / total_samples)
    return averaged_weights
