# main.py
import copy
import os
import time
import numpy as np

import torch
from tensorboardX import SummaryWriter

from tqdm import tqdm
import config
from update import LocalUpdate, test_inference
from models import CNNGarbage
from utils import get_dataset, average_weights

if __name__ == '__main__':
    # funzione per calcolare il tempo di esecuzione
    start_time = time.time()

    # path del progetto e funzione per trascrittura dei log
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    # configurazione iniziale del modello
    print('\nExperimental details:')
    print(f'    Model     : {config.MODEL}')
    print(f'    Optimizer : {config.OPTIMIZER}')
    print(f'    Learning  : {config.LR}')
    print(f'    Global Rounds   : {config.EPOCHS}\n')

    print('    Federated parameters:')
    print('    Non-IID')
    print(f'    Fraction of users  : {config.FRAC}')
    print(f'    Local Batch size   : {config.LOCAL_BS}')
    print(f'    Local Epochs       : {config.LOCAL_EP}\n')

    # device utilizzato per elaborazione dei dati
    device = config.CPU

    # Caricamento dataset e user groups.
    # Qui avviene la suddivisione del dataset in train (80%) e test (20%)
    # User_groups è l'insieme degli indici del train dataset su cui si lavorerà
    train_dataset, test_dataset, user_groups = get_dataset()

    # caricamento modello globale basato su CNN.
    # DATASET_CLASSES rappresenta il numero di classi utilizzate all'interno della CNN (1. garbage, 2. garbage_no)
    global_model = CNNGarbage(config.DATASET_CLASSES)

    # Imposta il modello da addestrare e lo invia al dispositivo
    global_model.to(device)
    global_model.train()
    print(global_model)

    # La funzione state_dict() restituisce i pesi del modello globale che vengono assegnati alla variabile global_weights
    global_weights = global_model.state_dict()

    # Fase di Training

    # Impostazione iniziale delle variabili per il training
    train_loss, train_accuracy = [], []
    print_every = 2

    # Iterazioni per 10 epoche
    for epoch in tqdm(range(config.EPOCHS)):
        local_weights, local_losses = [], []
        sample_counts = []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()  # Iniziamo il training tramite funzione apposita

        m = max(int(config.FRAC * config.NUM_USERS), 1)  # Calcola il numero di utenti da selezionare
        idxs_users = np.random.choice(range(config.NUM_USERS), m, replace=False)  # Seleziona m utenti casuali dall'insieme totale di utenti

        c = 0
        # Itera per ogni utente selezionato randomicamente
        for idx in idxs_users:
            # Richiamo il modello locale
            local_model = LocalUpdate(dataset=train_dataset, idxs=user_groups[idx], logger=logger)

            # Calcola l'accuracy iniziale per verificare se il modello è da aggiornare oppure no
            acc, _ = local_model.inference(model=global_model)
            c = c + 1
            print(f"User {idx} [{c}/{len(idxs_users)}] initial accuracy: {acc * 100:.2f}%")

            # Se l'accuracy è <= 70%, aggiorna i pesi locali
            if acc <= 0.7:
                w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
            else:
                local_weights.append(global_model.state_dict())
                local_losses.append(0)

            sample_counts.append(len(user_groups[idx]))  # Aggiungi il numero di campioni del client

        # Aggiorna i pesi globali se sono presenti pesi locali
        if local_weights:
            global_weights = average_weights(local_weights, sample_counts)
            global_model.load_state_dict(global_weights)

        # Calcolo la perdita media
        loss_avg = sum(local_losses) / len(local_losses) if local_losses else 0
        train_loss.append(loss_avg)

        # Calcola la training accuracy media su tutti gli utenti in ogni epoca
        list_acc, list_loss = [], []
        global_model.eval()  # Imposta il modello in modalità di valutazione
        for c in range(config.NUM_USERS):
            local_model = LocalUpdate(dataset=train_dataset, idxs=user_groups[c], logger=logger)  # Ottiene il modello locale
            acc, loss = local_model.inference(model=global_model)  # Calcola accuracy e loss
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # stampa la global training loss dopo ogni round 'i'
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    # Fase di Test

    # Inference Test dopo il completamento del training
    test_acc, test_loss = test_inference(global_model, test_dataset)

    # Stampa i risultati
    print(f' \n Results after {config.EPOCHS} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
