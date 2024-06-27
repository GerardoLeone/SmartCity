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
    start_time = time.time()

    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

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

    device = config.CPU

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset()

    global_model = CNNGarbage(config.DATASET_CLASSES)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    #Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # 10 epoch
    for epoch in tqdm(range(config.EPOCHS)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(config.FRAC * config.NUM_USERS), 1)
        idxs_users = np.random.choice(range(config.NUM_USERS), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.NUM_USERS):
            local_model = LocalUpdate(dataset=train_dataset,idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(global_model, test_dataset)

    print(f' \n Results after {config.EPOCHS} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
