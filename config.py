# Parametri utilizzati per addestramento del modello

EPOCHS = 10  # Numero totale di epoche per l'addestramento del modello
NUM_USERS = 100  # Numero totale di utenti che partecipano all'apprendimento federato
FRAC = 0.1  # Fractions of users to be sampled for each training round (10%)
LOCAL_EP = 10  # Numero di epoche locali di addestramento per ogni utente
LOCAL_BS = 10  # Dimensione del batch per l'addestramento locale
LR = 0.01  # Tasso di apprendimento (learning rate) per l'ottimizzazione
MOMENTUM = 0.5  # Momentum utilizzato nell'ottimizzatore

MODEL = 'cnn'  # Tipo di modello da utilizzare, in questo caso una rete neurale convoluzionale
NORM = 'batch_norm'  # Tipo di normalizzazione da applicare, in questo caso normalizzazione per batch

DATASET = 'garbage_classification'  # Nome del dataset utilizzato per l'addestramento
OPTIMIZER = 'sgd'  # Tipo di ottimizzatore da utilizzare, in questo caso Stochastic Gradient Descent (SGD)

CPU = 'cpu'  # Indica l'uso della CPU per l'addestramento
CUDA = 'cuda'  # Indica l'uso della GPU (CUDA) per l'addestramento

VERBOSE = 1  # Livello di verbosità per il logging durante l'addestramento (1 attivo, 0 inattivo)

# Dataset Const
DATASET_CLASSES = 2  # Numero di classi presenti nel dataset (ad esempio, per classificazione binaria)
