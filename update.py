#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config


# Classe per gestire un sottodataset basato su un insieme di indici specificati
class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        # Inizializza il sottodataset con il dataset originale e una lista di indici
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        # Restituisce il numero di campioni nel sottodataset
        return len(self.idxs)

    def __getitem__(self, item):
        # Restituisce l'immagine e l'etichetta corrispondenti all'indice fornito
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


# Classe per gestire l'aggiornamento locale dei pesi del modello in un contesto di apprendimento federato
class LocalUpdate(object):
    def __init__(self, dataset, idxs, logger):
        self.logger = logger
        # Divide il dataset negli insiemi di train, validazione e test
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = config.CPU
        # Imposta la funzione di perdita (criterion) su NLLLoss
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Restituisce i dataloader per il training, la validazione e il test
        per un dataset dato e un insieme di indici utente.
        """
        # Divide gli indici in 80% per training, 10% per validazione e 10% per test
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        # Crea i DataLoader per training, validazione e test
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=config.LOCAL_BS, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val) / 10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test) / 10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Imposta il modello in modalità di training
        model.train()
        epoch_loss = []

        # Imposta l'ottimizzatore per gli aggiornamenti locali
        optimizer = torch.optim.SGD(model.parameters(), lr=config.LR, momentum=0.5, weight_decay=1e-4)

        for iter in range(config.LOCAL_EP):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                # Sposta i dati su dispositivo (CPU o GPU)
                images, labels = images.to(self.device), labels.to(self.device)

                # Azzeramento dei gradienti del modello
                model.zero_grad()
                # Calcola le probabilità logaritmiche
                log_probs = model(images)
                # Calcola la perdita
                loss = self.criterion(log_probs, labels)
                # Backpropagation per calcolare i gradienti
                loss.backward()
                # Aggiorna i pesi del modello
                optimizer.step()

                if config.VERBOSE:
                    # Stampa informazioni sul training
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                # Aggiunge la perdita al logger
                self.logger.add_scalar('loss', loss.item())
                # Aggiunge la perdita del batch alla lista delle perdite
                batch_loss.append(loss.item())
            # Aggiunge la perdita media dell'epoca alla lista delle perdite delle epoche
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # Restituisce lo stato del modello e la perdita media dell'epoca
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Restituisce l'accuratezza e la perdita dell'inferenza.
        """

        # Imposta il modello in modalità di valutazione
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            # Sposta i dati su dispositivo (CPU o GPU)
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            # Calcola la perdita del batch
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Predizione
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            # Calcola il numero di predizioni corrette
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        # Calcola l'accuratezza
        accuracy = correct / total
        return accuracy, loss


# Funzione per calcolare l'accuratezza e la perdita su un dataset di test
def test_inference(model, test_dataset):
    """ Restituisce l'accuratezza e la perdita del test.
    """

    # Imposta il modello in modalità di valutazione
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = config.CPU
    # Imposta la funzione di perdita su NLLLoss
    criterion = nn.NLLLoss().to(device)
    # Crea un DataLoader per il dataset di test
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        # Sposta i dati su dispositivo (CPU o GPU)
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        # Calcola la perdita del batch
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Predizione
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        # Calcola il numero di predizioni corrette
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    # Calcola l'accuratezza
    accuracy = correct / total
    return accuracy, loss
