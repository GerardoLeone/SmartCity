import copy
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

import config


def get_dataset(train_ratio=0.8):
    # Trasformazioni da applicare alle immagini
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Ridimensiona le immagini a 128x128 pixel
        transforms.RandomRotation(30),  # Ruota casualmente le immagini fino a 30 gradi
        transforms.RandomHorizontalFlip(),  # Applica flip orizzontale casuale
        transforms.RandomVerticalFlip(),  # Applica flip verticale casuale
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # Modifica casualmente le proprietà di colore
        transforms.ToTensor(),  # Converte le immagini in tensori
        transforms.Normalize((0.5,), (0.5,)),  # Normalizza i tensori
        transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x))  # Aggiunge rumore gaussiano
    ])

    # Carica il dataset dalle immagini nella cartella 'garbage_classification'
    dataset = datasets.ImageFolder(root='garbage_classification/', transform=transform)

    # Suddivisione del dataset in training set e test set
    train_size = int(train_ratio * len(dataset))  # Calcola la dimensione del training set
    test_size = len(dataset) - train_size  # Calcola la dimensione del test set
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # Divide il dataset

    # Suddivisione dei dati del training set per utente
    user_group = {}
    indices = list(range(len(train_dataset)))  # Crea una lista di indici
    split = len(indices) // config.NUM_USERS  # Calcola la dimensione del blocco per utente

    for i in range(config.NUM_USERS):
        start_idx = i * split  # Inizio del blocco
        end_idx = (i + 1) * split if i != config.NUM_USERS - 1 else len(indices)  # Fine del blocco
        user_group[i] = indices[start_idx:end_idx]  # Assegna gli indici all'utente

    return train_dataset, test_dataset, user_group  # Restituisce i dataset e il gruppo di utenti


def average_weights(w, sample_counts):
    """
    Media i pesi usando i conteggi dei campioni come pesi.
    Args:
      w: Una lista di state_dict da diversi modelli.
      sample_counts: Una lista di interi che rappresentano il numero di campioni per ciascun client.
    Returns:
      Il state_dict mediato.
    """
    total_samples = sum(sample_counts)  # Calcola il numero totale di campioni
    averaged_weights = copy.deepcopy(w[0])  # Crea una copia dei pesi del primo modello
    for key in averaged_weights.keys():
        for i in range(len(w)):
            if i == 0:
                averaged_weights[key] = w[i][key] * (sample_counts[i] / total_samples)  # Inizializza la media pesata
            else:
                averaged_weights[key] += w[i][key] * (sample_counts[i] / total_samples)  # Aggiunge il contributo pesato
    return averaged_weights  # Restituisce i pesi mediati
