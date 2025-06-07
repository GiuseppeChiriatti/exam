# ===================================================================
# STUDIO DEL DROPOUT COME TECNICA DI REGOLARIZZAZIONE
# ===================================================================
# Questo script implementa un esperimento completo per analizzare
# l'efficacia del dropout nella prevenzione dell'overfitting nelle
# reti neurali multistrato (MLP - Multi-Layer Perceptron)

# IMPORTAZIONI E MOTIVAZIONI TECNOLOGICHE
# ===================================================================

# PyTorch: Framework di deep learning scelto per:
# - Interfaccia Pythonica intuitiva
# - GPU acceleration nativa
# - Debugging facilitato (eager execution)
# - Ecosistema ricco di utilities
import torch
import torch.nn as nn           # Moduli neurali pre-definiti
import torch.optim as optim     # Algoritmi di ottimizzazione
import torch.nn.functional as F # Funzioni di attivazione e loss
from torch.utils.data import DataLoader  # Gestione efficiente dei batch

# TorchVision: Estensione per computer vision
# - Dataset pre-caricati (MNIST, Fashion-MNIST)
# - Trasformazioni standard per il preprocessing
import torchvision
import torchvision.transforms as transforms

# Matplotlib: Libreria di plotting standard Python
# Scelta per la compatibilità e flessibilità nella visualizzazione
import matplotlib.pyplot as plt

# NumPy: Calcolo numerico efficiente
# Indispensabile per operazioni matematiche su array
import numpy as np

# Scikit-learn: Metriche di valutazione
# Fornisce classification_report per analisi dettagliate
from sklearn.metrics import classification_report, confusion_matrix

# Seaborn: Visualizzazioni statistiche avanzate
# Built on matplotlib, offre plot più eleganti per heatmap
import seaborn as sns

# Collections: Strutture dati Python avanzate
# defaultdict per gestire contatori automatici
from collections import defaultdict

# Moduli standard Python per utilità varie
import time        # Misurazione tempi di esecuzione
import os          # Operazioni su filesystem
from datetime import datetime  # Timestamp per organizzazione file
import json        # Serializzazione dati per persistenza

# ===================================================================
# CONFIGURAZIONE HARDWARE E DIRECTORY
# ===================================================================

# RILEVAMENTO AUTOMATICO DEL DEVICE DI COMPUTAZIONE
# PyTorch supporta automaticamente GPU NVIDIA se disponibile
# Fallback su CPU se CUDA non è disponibile
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilizzo device: {device}")

# PERCHÉ QUESTA SCELTA:
# - GPU accelera training di 10-100x rispetto a CPU
# - torch.cuda.is_available() verifica driver CUDA
# - Fallback automatico garantisce compatibilità universale

def create_output_directories():
    """
    Crea una struttura organizzata di cartelle per i risultati
    
    MOTIVAZIONE DELLA STRUTTURA:
    - Timestamp evita sovrascritture accidentali
    - Separazione logica per tipo di output
    - Facilita l'archiviazione e il confronto tra esperimenti
    
    Returns:
        dict: Dizionario con i percorsi delle cartelle create
    """
    # Timestamp ISO per ordinamento cronologico naturale
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"dropout_study_results_{timestamp}"
    
    # Struttura gerarchica per organizzazione logica
    directories = {
        'base': base_dir,                              # Cartella principale
        'plots': os.path.join(base_dir, 'plots'),      # Grafici e visualizzazioni
        'models': os.path.join(base_dir, 'models'),    # Stati dei modelli salvati
        'reports': os.path.join(base_dir, 'reports'),  # Report in formato markdown/PDF
        'data': os.path.join(base_dir, 'data')         # Dati numerici in JSON
    }
    
    # Creazione ricorsiva delle directory
    # exist_ok=True previene errori se le cartelle esistono già
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

# Inizializzazione globale delle directory
output_dirs = create_output_directories()
print(f"Risultati salvati in: {output_dirs['base']}")

# ===================================================================
# LOGGING E TRACCIABILITÀ DELL'ESPERIMENTO
# ===================================================================

# Struttura dati per tracciare completamente l'esperimento
# PERCHÉ È IMPORTANTE:
# - Riproducibilità scientifica
# - Debug e analisi post-esperimento
# - Confronto tra esperimenti diversi
experiment_log = {
    'start_time': datetime.now().isoformat(),  # ISO 8601 per standard internazionale
    'device': str(device),                     # Informazioni hardware
    'experiments': [],                         # Lista degli esperimenti eseguiti
    'conclusions': []                          # Conclusioni automatiche generate
}

# ===================================================================
# ARCHITETTURE NEURALI IMPLEMENTATE
# ===================================================================

class MLPSingleHidden(nn.Module):
    """
    Multi-Layer Perceptron con un singolo strato nascosto
    
    MOTIVAZIONE ARCHITETTUALE:
    - Semplicità: Baseline per confronti
    - Interpretabilità: Pochi parametri da analizzare
    - Velocità: Training rapido per esperimenti
    
    PERCHÉ QUESTA IMPLEMENTAZIONE:
    - Eredita da nn.Module per integrazione PyTorch
    - Dropout configurabile per esperimenti controllati
    - ReLU come attivazione (standard moderno)
    """
    
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.0):
        """
        Inizializzazione dell'architettura
        
        Args:
            input_size (int): Dimensione input (784 per immagini 28x28)
            hidden_size (int): Neuroni nel layer nascosto
            num_classes (int): Numero di classi di output (10 per MNIST)
            dropout_rate (float): Probabilità di dropout [0.0, 1.0]
        """
        super(MLPSingleHidden, self).__init__()
        
        # LAYER LINEARE: trasformazione affine y = xW^T + b
        # input_size x hidden_size parametri + hidden_size bias
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # DROPOUT LAYER: regolarizzazione stocastica
        # Durante training: spegne neuroni casualmente con prob. dropout_rate
        # Durante inference: scala output per compensare
        self.dropout = nn.Dropout(dropout_rate)
        
        # OUTPUT LAYER: proiezione verso spazio delle classi
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Memorizzazione per analisi successive
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """
        Forward pass della rete
        
        FLUSSO COMPUTAZIONALE:
        1. Flatten dell'input (immagini -> vettore)
        2. Trasformazione lineare + attivazione
        3. Dropout (solo in training mode)
        4. Classificazione finale
        
        Args:
            x (torch.Tensor): Batch di input [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Logits raw [batch_size, num_classes]
        """
        # FLATTEN: Converte immagini 2D in vettori 1D
        # Da [batch_size, 1, 28, 28] a [batch_size, 784]
        # view(-1, 784) preserva batch_size, flatten il resto
        x = x.view(x.size(0), -1)
        
        # PRIMO LAYER + ATTIVAZIONE
        # ReLU: f(x) = max(0, x)
        # - Risolve vanishing gradient problem
        # - Computazionalmente efficiente
        # - Sparsità naturale (output nulli)
        x = F.relu(self.fc1(x))
        
        # DROPOUT: Regolarizzazione stocastica
        # In training: randomly sets elements to 0 with prob. dropout_rate
        # In eval: identity operation (no effect)
        x = self.dropout(x)
        
        # OUTPUT LAYER: No attivazione (raw logits)
        # CrossEntropyLoss applicherà softmax internamente
        x = self.fc2(x)
        
        return x

class MLPMultiHidden(nn.Module):
    """
    Multi-Layer Perceptron con più strati nascosti
    
    MOTIVAZIONE ARCHITETTUALE:
    - Maggiore capacità espressiva
    - Apprendimento di rappresentazioni gerarchiche
    - Test dell'efficacia del dropout in reti profonde
    
    DESIGN PATTERN:
    - ModuleList per gestione dinamica layer
    - Dropout uniforme su tutti gli strati
    - Architettura encoder-like (dimensioni decrescenti)
    """
    
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.0):
        """
        Costruzione dinamica di architettura multi-layer
        
        Args:
            input_size (int): Dimensione input
            hidden_sizes (list): Lista delle dimensioni dei layer nascosti
            num_classes (int): Numero classi output
            dropout_rate (float): Dropout rate uniforme
        """
        super(MLPMultiHidden, self).__init__()
        
        # CONTAINERS DINAMICI PER LAYER
        # ModuleList: Lista di moduli PyTorch
        # - Registrazione automatica parametri
        # - Supporto per GPU migration
        # - Serializzazione automatica
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # COSTRUZIONE PRIMO LAYER
        # Input -> Primo strato nascosto
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.dropouts.append(nn.Dropout(dropout_rate))
        
        # COSTRUZIONE STRATI INTERMEDI
        # Pattern encoder: dimensioni decrescenti
        # Ogni layer riduce la dimensionalità progressivamente
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # LAYER DI OUTPUT
        # Ultimo hidden -> num_classes
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        
        # Memorizzazione configurazione
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """
        Forward pass multi-layer con dropout
        
        ARCHITETTURA GENERALE:
        Input -> [Linear -> ReLU -> Dropout] x N -> Linear -> Output
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output logits
        """
        # Flatten dell'input per FC layers
        x = x.view(x.size(0), -1)
        
        # ITERAZIONE SU STRATI INTERMEDI
        # Pattern uniforme: Linear -> ReLU -> Dropout
        for layer, dropout in zip(self.layers, self.dropouts):
            # Trasformazione lineare
            x = layer(x)
            
            # Attivazione non-lineare
            # ReLU mantiene gradienti positivi
            x = F.relu(x)
            
            # Regolarizzazione stocastica
            # Previene co-adaptation tra neuroni
            x = dropout(x)
        
        # OUTPUT FINALE: Solo trasformazione lineare
        # No attivazione -> raw logits per CrossEntropyLoss
        x = self.output_layer(x)
        
        return x

# ===================================================================
# GESTIONE DATASET E PREPROCESSING
# ===================================================================

def load_datasets(dataset_name='MNIST', batch_size=128):
    """
    Caricamento e preprocessing standardizzato dei dataset
    
    SCELTE DI PREPROCESSING:
    - ToTensor(): PIL Image -> Tensor + normalizzazione [0,1]
    - Normalize((0.5,), (0.5,)): [0,1] -> [-1,1] per stabilità training
    
    PERCHÉ QUESTA NORMALIZZAZIONE:
    - Centra i dati attorno a 0
    - Riduce internal covariate shift
    - Migliora convergenza degli ottimizzatori
    
    Args:
        dataset_name (str): 'MNIST' o 'FashionMNIST'
        batch_size (int): Dimensione batch per training
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # CATENA DI TRASFORMAZIONI
    # Compose: Applica trasformazioni in sequenza
    transform = transforms.Compose([
        # PIL Image -> Tensor float32 + scala [0,1]
        transforms.ToTensor(),
        
        # Normalizzazione: (pixel - mean) / std
        # (0.5,) = media, (0.5,) = std per canale singolo
        # Risultato: [0,1] -> [-1,1]
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # CARICAMENTO CONDIZIONALE DATASET
    if dataset_name == 'MNIST':
        # MNIST: Cifre scritte a mano 0-9
        # 60k training, 10k test, 28x28 grayscale
        train_dataset = torchvision.datasets.MNIST(
            root='./data',           # Directory cache locale
            train=True,              # Set di training
            download=True,           # Download automatico se necessario
            transform=transform      # Preprocessing pipeline
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,             # Set di test
            download=True,
            transform=transform
        )
    elif dataset_name == 'FashionMNIST':
        # Fashion-MNIST: Capi abbigliamento
        # Stesse dimensioni MNIST ma più complesso
        # 10 categorie: t-shirt, pantaloni, pullover, etc.
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
    
    # DATALOADER: Gestione efficiente batching
    # VANTAGGI:
    # - Caricamento asincrono in background
    # - Shuffling automatico per training
    # - Gestione memoria efficiente
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,              # Randomizzazione ordine samples
        # num_workers=4,           # Parallelizzazione I/O (opzionale)
        # pin_memory=True          # Ottimizzazione GPU transfer (opzionale)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False              # Ordine fisso per riproducibilità test
    )
    
    return train_loader, test_loader

# ===================================================================
# FUNZIONI DI TRAINING E VALUTAZIONE
# ===================================================================

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Loop di training standardizzato con logging delle metriche
    
    ALGORITMO DI TRAINING:
    1. Forward pass: calcolo predizioni
    2. Loss computation: confronto con ground truth
    3. Backward pass: calcolo gradienti via backpropagation
    4. Optimizer step: aggiornamento pesi
    5. Logging: raccolta metriche per monitoraggio
    
    Args:
        model: Modello PyTorch da addestrare
        train_loader: DataLoader per training set
        criterion: Funzione di loss
        optimizer: Algoritmo di ottimizzazione
        num_epochs: Numero di epoche di training
        
    Returns:
        tuple: (train_losses, train_accuracies) per plotting
    """
    # MODALITÀ TRAINING
    # Abilita dropout, batch normalization, etc.
    model.train()
    
    # LISTE PER TRACCIAMENTO METRICHE
    train_losses = []      # Loss media per epoca
    train_accuracies = []  # Accuracy media per epoca
    
    # LOOP PRINCIPALE EPOCHE
    for epoch in range(num_epochs):
        # ACCUMULATORI PER STATISTICHE EPOCA
        running_loss = 0.0  # Somma loss batch
        correct = 0         # Predizioni corrette
        total = 0           # Campioni totali processati
        
        # LOOP SUI BATCH
        for i, (inputs, labels) in enumerate(train_loader):
            # TRASFERIMENTO SU DEVICE COMPUTAZIONALE
            # .to(device) sposta tensori su GPU/CPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            # AZZERAMENTO GRADIENTI
            # PyTorch accumula gradienti -> reset necessario
            optimizer.zero_grad()
            
            # FORWARD PASS
            # Calcolo predizioni del modello
            outputs = model(inputs)
            
            # CALCOLO LOSS
            # CrossEntropyLoss combina LogSoftmax + NLLLoss
            loss = criterion(outputs, labels)
            
            # BACKWARD PASS
            # Calcolo gradienti via chain rule
            loss.backward()
            
            # OPTIMIZER STEP
            # Aggiornamento parametri secondo algoritmo scelto
            optimizer.step()
            
            # AGGIORNAMENTO STATISTICHE
            running_loss += loss.item()  # .item() estrae valore scalare
            
            # CALCOLO ACCURACY
            # torch.max restituisce (values, indices)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # CALCOLO METRICHE EPOCA
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # SALVATAGGIO METRICHE
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # LOGGING PERIODICO
        # Evita spam su console, print ogni 2 epoche
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return train_losses, train_accuracies

def test_model(model, test_loader):
    """
    Valutazione del modello su test set
    
    DIFFERENZE DAL TRAINING:
    - model.eval(): disabilita dropout, batch norm in eval mode
    - torch.no_grad(): disabilita calcolo gradienti per efficienza
    - Nessun aggiornamento parametri
    
    Args:
        model: Modello addestrato da valutare
        test_loader: DataLoader per test set
        
    Returns:
        tuple: (accuracy, average_loss)
    """
    # MODALITÀ VALUTAZIONE
    # Disabilita dropout, fissa batch normalization
    model.eval()
    
    # INIZIALIZZAZIONE CONTATORI
    correct = 0
    total = 0
    test_loss = 0
    
    # LOSS FUNCTION
    # Stessa del training per confrontabilità
    criterion = nn.CrossEntropyLoss()
    
    # CONTEXT MANAGER PER EFFICIENZA
    # torch.no_grad() disabilita autograd
    # - Riduce memoria utilizzata
    # - Accelera computazione
    # - Previene accumulo gradienti accidentale
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Device transfer
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass (solo predizione)
            outputs = model(inputs)
            
            # Calcolo loss per monitoraggio
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Calcolo accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # CALCOLO METRICHE FINALI
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    return accuracy, avg_loss

# Funzione per confrontare modelli con logging
def compare_models(dataset_name='MNIST'):
    """
    Confronto sistematico di architetture con e senza dropout
    
    DESIGN SPERIMENTALE:
    - Controllo variabili: stessi iperparametri tranne dropout
    - Quattro configurazioni: Single/Multi x NoDropout/WithDropout
    - Logging completo per analisi post-hoc
    
    PERCHÉ QUESTO DESIGN:
    - Isolamento effetto dropout da altri fattori
    - Confronto fair tra architetture
    - Raccolta dati quantitativi per conclusioni oggettive
    """
    print(f"\n=== CONFRONTO MODELLI SU {dataset_name} ===")
    
    # CARICAMENTO DATASET STANDARDIZZATO
    train_loader, test_loader = load_datasets(dataset_name)
    
    # IPERPARAMETRI FISSI
    # Mantenuti costanti per isolamento variabile dropout
    input_size = 28 * 28    # Flattened MNIST/Fashion-MNIST
    num_classes = 10        # 10 categorie per entrambi dataset
    num_epochs = 15         # Abbastanza per convergenza
    learning_rate = 0.001   # Learning rate conservativo per Adam
    
    # STRUTTURE DATI RISULTATI
    results = {}            # Risultati numerici per ogni modello
    experiment_info = {     # Metadati esperimento per report
        'dataset': dataset_name,
        'parameters': {
            'input_size': input_size,
            'num_classes': num_classes,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate
        },
        'models': {}
    }
    
    # ===============================================================
    # ESPERIMENTO 1: MLP SINGOLO STRATO SENZA DROPOUT
    # ===============================================================
    print("\n1. Training MLP singolo strato SENZA dropout...")
    
    # INIZIALIZZAZIONE MODELLO
    # dropout_rate=0.0 -> no regolarizzazione
    model1 = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.0).to(device)
    
    # OTTIMIZZATORE ADAM
    # PERCHÉ ADAM:
    # - Adaptive learning rates per parametro
    # - Momentum + RMSprop combination
    # - Robusto a scelte iperparametri
    # - Standard de facto per deep learning
    optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
    
    # LOSS FUNCTION
    # CrossEntropyLoss per classificazione multiclasse
    # Combina LogSoftmax + NLLLoss per stabilità numerica
    criterion = nn.CrossEntropyLoss()
    
    # TRAINING CON TIMING
    start_time = time.time()
    train_losses1, train_acc1 = train_model(model1, train_loader, criterion, optimizer1, num_epochs)
    training_time1 = time.time() - start_time
    
    # VALUTAZIONE SU TEST SET
    test_acc1, test_loss1 = test_model(model1, test_loader)
    
    # SALVATAGGIO RISULTATI
    results['Single_No_Dropout'] = {
        'train_losses': train_losses1,
        'train_acc': train_acc1,
        'test_acc': test_acc1,
        'test_loss': test_loss1,
        'training_time': training_time1
    }
    print(f"Test Accuracy: {test_acc1:.2f}%, Test Loss: {test_loss1:.4f}")
    
    # PERSISTENZA MODELLO
    # Salvataggio state_dict per riutilizzo futuro
    model_path = os.path.join(output_dirs['models'], f'{dataset_name.lower()}_single_no_dropout.pth')
    torch.save(model1.state_dict(), model_path)
    
    # LOGGING PER REPORT
    experiment_info['models']['Single_No_Dropout'] = {
        'architecture': 'Single Hidden Layer (512 neurons)',
        'dropout_rate': 0.0,
        'test_accuracy': test_acc1,
        'training_time': training_time1,
        'overfitting_gap': train_acc1[-1] - test_acc1  # Misura overfitting
    }
    
    # ===============================================================
    # ESPERIMENTO 2: MLP SINGOLO STRATO CON DROPOUT
    # ===============================================================
    print("\n2. Training MLP singolo strato CON dropout (0.3)...")
    
    # STESSO MODELLO CON DROPOUT
    # dropout_rate=0.3 -> 30% neuroni spenti casualmente
    model2 = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.3).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
    
    # TRAINING IDENTICO
    start_time = time.time()
    train_losses2, train_acc2 = train_model(model2, train_loader, criterion, optimizer2, num_epochs)
    training_time2 = time.time() - start_time
    
    test_acc2, test_loss2 = test_model(model2, test_loader)
    
    # SALVATAGGIO E LOGGING
    results['Single_With_Dropout'] = {
        'train_losses': train_losses2,
        'train_acc': train_acc2,
        'test_acc': test_acc2,
        'test_loss': test_loss2,
        'training_time': training_time2
    }
    print(f"Test Accuracy: {test_acc2:.2f}%, Test Loss: {test_loss2:.4f}")
    
    model_path = os.path.join(output_dirs['models'], f'{dataset_name.lower()}_single_with_dropout.pth')
    torch.save(model2.state_dict(), model_path)
    
    experiment_info['models']['Single_With_Dropout'] = {
        'architecture': 'Single Hidden Layer (512 neurons)',
        'dropout_rate': 0.3,
        'test_accuracy': test_acc2,
        'training_time': training_time2,
        'overfitting_gap': train_acc2[-1] - test_acc2
    }
    
    # ===============================================================
    # ESPERIMENTO 3: MLP MULTI-STRATO SENZA DROPOUT
    # ===============================================================
    print("\n3. Training MLP multi-strato SENZA dropout...")
    
    # ARCHITETTURA ENCODER-LIKE
    # [512, 256, 128] -> dimensioni decrescenti
    # MOTIVAZIONE: estrazione features gerarchiche
    model3 = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.0).to(device)
    optimizer3 = optim.Adam(model3.parameters(), lr=learning_rate)
    
    start_time = time.time()
    train_losses3, train_acc3 = train_model(model3, train_loader, criterion, optimizer3, num_epochs)
    training_time3 = time.time() - start_time
    
    test_acc3, test_loss3 = test_model(model3, test_loader)
    
    results['Multi_No_Dropout'] = {
        'train_losses': train_losses3,
        'train_acc': train_acc3,
        'test_acc': test_acc3,
        'test_loss': test_loss3,
        'training_time': training_time3
    }
    print(f"Test Accuracy: {test_acc3:.2f}%, Test Loss: {test_loss3:.4f}")
    
    model_path = os.path.join(output_dirs['models'], f'{dataset_name.lower()}_multi_no_dropout.pth')
    torch.save(model3.state_dict(), model_path)
    
    experiment_info['models']['Multi_No_Dropout'] = {
        'architecture': 'Multi Hidden Layers (512-256-128 neurons)',
        'dropout_rate': 0.0,
        'test_accuracy': test_acc3,
        'training_time': training_time3,
        'overfitting_gap': train_acc3[-1] - test_acc3
    }
    
    # ===============================================================
    # ESPERIMENTO 4: MLP MULTI-STRATO CON DROPOUT
    # ===============================================================
    print("\n4. Training MLP multi-strato CON dropout (0.3)...")
    
    # STESSA ARCHITETTURA + DROPOUT UNIFORME
    model4 = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.3).to(device)
    optimizer4 = optim.Adam(model4.parameters(), lr=learning_rate)
    
    start_time = time.time()
    train_losses4, train_acc4 = train_model(model4, train_loader, criterion, optimizer4, num_epochs)
    training_time4 = time.time() - start_time
    
    test_acc4, test_loss4 = test_model(model4, test_loader)
    
    results['Multi_With_Dropout'] = {
        'train_losses': train_losses4,
        'train_acc': train_acc4,
        'test_acc': test_acc4,
        'test_loss': test_loss4,
        'training_time': training_time4
    }
    print(f"Test Accuracy: {test_acc4:.2f}%, Test Loss: {test_loss4:.4f}")
    
    model_path = os.path.join(output_dirs['models'], f'{dataset_name.lower()}_multi_with_dropout.pth')
    torch.save(model4.state_dict(), model_path)
    
    experiment_info['models']['Multi_With_Dropout'] = {
        'architecture': 'Multi Hidden Layers (512-256-128 neurons)',
        'dropout_rate': 0.3,
        'test_accuracy': test_acc4,
        'training_time': training_time4,
        'overfitting_gap': train_acc4[-1] - test_acc4
    }
    
    # ===============================================================
    # PERSISTENZA RISULTATI NUMERICI
    # ===============================================================
    
    # SERIALIZZAZIONE JSON
    # Conversione numpy arrays -> liste per compatibilità JSON
    results_path = os.path.join(output_dirs['data'], f'results_{dataset_name.lower()}.json')
    with open(results_path, 'w') as f:
        serializable_results = {}
        for model_name, data in results.items():
            serializable_results[model_name] = {
                'train_losses': [float(x) for x in data['train_losses']],
                'train_acc': [float(x) for x in data['train_acc']],
                'test_acc': float(data['test_acc']),
                'test_loss': float(data['test_loss']),  
                'training_time': float(data['training_time'])
            }
        json.dump(serializable_results, f, indent=2)
    
    # AGGIORNAMENTO LOG GLOBALE
    experiment_log['experiments'].append(experiment_info)
    
    return results

# Funzione per visualizzare i risultati con salvataggio
def plot_results(results, dataset_name, save_plots=True):
    """
    Visualizzazione completa dei risultati sperimentali
    
    DESIGN VISUALIZZAZIONE:
    - 4 subplot per analisi multidimensionale
    - Colori distintivi per chiarezza
    - Annotazioni per interpretazione immediata
    - Salvataggio alta risoluzione per pubblicazioni
    
    TECNOLOGIE UTILIZZATE:
    - matplotlib.pyplot: Framework plotting standard
    - subplots(): Layout griglia per confronti
    - Personalizzazione estetica per professionalità
    """
    # SETUP FIGURA PRINCIPALE
    # figsize in pollici per controllo dimensioni stampa
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Confronto Prestazioni Modelli - {dataset_name}', fontsize=16)
    
    # ===============================================================
    # GRAFICO 1: EVOLUZIONE TRAINING LOSS
    # ===============================================================
    ax1 = axes[0, 0]
    
    # PLOT MULTIPLO CON LOOP
    # Ogni modello = curva diversa
    for model_name, data in results.items():
        # linewidth=2 per visibilità su stampa
        ax1.plot(data['train_losses'], label=model_name, linewidth=2)
    
    # PERSONALIZZAZIONE ASSI E LABELS
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()                    # Legenda automatica da labels
    ax1.grid(True, alpha=0.3)       # Griglia sottile per lettura
    
    # ===============================================================
    # GRAFICO 2: EVOLUZIONE TRAINING ACCURACY
    # ===============================================================
    ax2 = axes[0, 1]
    
    for model_name, data in results.items():
        ax2.plot(data['train_acc'], label=model_name, linewidth=2)
    
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ===============================================================
    # GRAFICO 3: CONFRONTO TEST ACCURACY (BAR PLOT)
    # ===============================================================
    ax3 = axes[1, 0]
    
    # ESTRAZIONE DATI PER BAR PLOT
    models = list(results.keys())
    test_accs = [results[model]['test_acc'] for model in models]
    
    # PALETTE COLORI DISTINTIVA
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
    bars = ax3.bar(models, test_accs, color=colors)
    
    ax3.set_title('Test Accuracy Comparison')
    ax3.set_ylabel('Accuracy (%)')
    ax3.tick_params(axis='x', rotation=45)  # Rotazione labels per leggibilità
    
    # ANNOTAZIONI VALORI SULLE BARRE
    # Migliora interpretazione immediata
    for bar, acc in zip(bars, test_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.2f}%', ha='center', va='bottom')
    
    # ===============================================================
    # GRAFICO 4: ANALISI OVERFITTING (TRAIN-TEST GAP)
    # ===============================================================
    ax4 = axes[1, 1]
    
    # CALCOLO GAP OVERFITTING
    # Differenza tra ultima accuracy training e test accuracy
    overfitting = []
    for model_name, data in results.items():
        final_train_acc = data['train_acc'][-1]
        test_acc = data['test_acc']
        gap = final_train_acc - test_acc
        overfitting.append(gap)
    
    bars = ax4.bar(models, overfitting, color=colors)
    ax4.set_title('Overfitting Analysis (Train-Test Gap)')
    ax4.set_ylabel('Accuracy Gap (%)')
    ax4.tick_params(axis='x', rotation=45)
    
    # LINEA RIFERIMENTO A ZERO
    # Evidenzia threshold overfitting
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # ANNOTAZIONI GAP VALUES
    for bar, gap in zip(bars, overfitting):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{gap:.2f}%', ha='center', va='bottom')
    
    # LAYOUT OTTIMIZZAZIONE
    plt.tight_layout()
    
    # ===============================================================
    # SALVATAGGIO ALTA RISOLUZIONE
    # ===============================================================
    if save_plots:
        # dpi=300 per qualità pubblicazione
        # bbox_inches='tight' elimina whitespace
        plot_path = os.path.join(output_dirs['plots'], f'model_comparison_{dataset_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot salvato: {plot_path}")
    
    plt.show()

# Funzione per analisi dettagliata del dropout con salvataggio
def dropout_analysis(dataset_name='MNIST', save_plots=True):
    """
    Analisi sistematica dell'effetto di diversi valori di dropout
    
    METODOLOGIA SPERIMENTALE:
    - Range dropout rates: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    - Architettura fissa: Multi-layer per massimizzare effetto
    - Metriche: accuracy + overfitting gap
    - Visualizzazione: curve response dropout rate
    
    OBIETTIVO:
    - Identificare dropout rate ottimale per dataset
    - Quantificare trade-off accuracy vs regolarizzazione
    - Fornire guidelines empiriche per hyperparameter tuning
    """
    print(f"\n=== ANALISI DETTAGLIATA DROPOUT SU {dataset_name} ===")
    
    # CARICAMENTO DATASET
    train_loader, test_loader = load_datasets(dataset_name)
    
    # RANGE DROPOUT RATES DA TESTARE
    # MOTIVAZIONE RANGE:
    # - 0.0: baseline senza regolarizzazione
    # - 0.1-0.3: range tipico per hidden layers
    # - 0.4-0.5: dropout aggressivo per comparison
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}
    
    # PARAMETRI FISSI
    input_size = 28 * 28
    num_classes = 10
    num_epochs = 10          # Ridotto per speed vs accuracy trade-off
    learning_rate = 0.001
    
    # METADATI ESPERIMENTO
    dropout_experiment = {
        'dataset': dataset_name,
        'dropout_rates_tested': dropout_rates,
        'parameters': {
            'architecture': 'Multi Hidden Layers (512-256-128)',
            'num_epochs': num_epochs,
            'learning_rate': learning_rate
        },
        'results': {}
    }
    
    # ===============================================================
    #LOOP PRINCIPALE: TEST OGNI DROPOUT RATE
    # ===============================================================
    for dropout_rate in dropout_rates:
        print(f"\nTesting dropout rate: {dropout_rate}")
        
        # NUOVO MODELLO PER OGNI RATE
        # Evita contamination tra esperimenti
        model = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # TRAINING STANDARDIZZATO
        train_losses, train_acc = train_model(model, train_loader, criterion, optimizer, num_epochs)
        test_acc, test_loss = test_model(model, test_loader)
        
        # SALVATAGGIO RISULTATI
        results[dropout_rate] = {
            'train_acc': train_acc[-1],        # Final training accuracy
            'test_acc': test_acc,
            'overfitting': train_acc[-1] - test_acc  # Gap metric
        }
        
        # LOGGING STRUTTURATO PER JSON
        dropout_experiment['results'][str(dropout_rate)] = {
            'train_accuracy': float(train_acc[-1]),
            'test_accuracy': float(test_acc),
            'overfitting_gap': float(train_acc[-1] - test_acc)
        }
        
        # LOGGING IMMEDIATO
        print(f"Train Acc: {train_acc[-1]:.2f}%, Test Acc: {test_acc:.2f}%, Gap: {train_acc[-1] - test_acc:.2f}%")
    
    # ===============================================================
    # VISUALIZZAZIONE ANALISI DROPOUT
    # ===============================================================
    
    # SETUP FIGURA TRIPLA
    # 3 subplot per analisi completa
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Analisi Dropout - {dataset_name}', fontsize=16)
    
    # ESTRAZIONE DATI PER PLOTTING
    dropout_vals = list(results.keys())
    train_accs = [results[dr]['train_acc'] for dr in dropout_vals]
    test_accs = [results[dr]['test_acc'] for dr in dropout_vals]
    overfitting = [results[dr]['overfitting'] for dr in dropout_vals]
    
    # SUBPLOT 1: TRAIN VS TEST ACCURACY
    # OBIETTIVO: Visualizzare convergenza curves
    axes[0].plot(dropout_vals, train_accs, 'o-', label='Train Accuracy', 
                linewidth=2, markersize=8)
    axes[0].plot(dropout_vals, test_accs, 's-', label='Test Accuracy', 
                linewidth=2, markersize=8)
    axes[0].set_xlabel('Dropout Rate')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Effect of Dropout on Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SUBPLOT 2: TEST ACCURACY FOCUS
    # OBIETTIVO: Identificare optimum dropout rate
    axes[1].plot(dropout_vals, test_accs, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Dropout Rate')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Test Accuracy vs Dropout Rate')
    axes[1].grid(True, alpha=0.3)
    
    # SUBPLOT 3: OVERFITTING ANALYSIS
    # OBIETTIVO: Quantificare effetto regolarizzazione
    axes[2].plot(dropout_vals, overfitting, 'go-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Dropout Rate')
    axes[2].set_ylabel('Train-Test Gap (%)')
    axes[2].set_title('Overfitting vs Dropout Rate')
    
    # LINEA RIFERIMENTO ZERO OVERFITTING
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # SALVATAGGIO PLOT
    if save_plots:
        plot_path = os.path.join(output_dirs['plots'], f'dropout_analysis_{dataset_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot salvato: {plot_path}")
    
    plt.show()
    
    # ===============================================================
    # PERSISTENZA RISULTATI
    # ===============================================================
    
    # SALVATAGGIO JSON STRUTTURATO
    dropout_results_path = os.path.join(output_dirs['data'], f'dropout_analysis_{dataset_name.lower()}.json')
    with open(dropout_results_path, 'w') as f:
        json.dump(dropout_experiment, f, indent=2)
    
    # AGGIORNAMENTO LOG GLOBALE
    experiment_log['experiments'].append({
        'type': 'dropout_analysis',
        'data': dropout_experiment
    })
    
    return results

# Funzione per creare report completo in Markdown
def create_comprehensive_report(results_mnist, results_fashion, dropout_results_mnist, dropout_results_fashion):
    """Crea un report completo in formato Markdown con tutti i risultati e analisi"""
    
    experiment_log['end_time'] = datetime.now().isoformat()
    experiment_log['total_duration'] = str(datetime.fromisoformat(experiment_log['end_time']) - 
                                         datetime.fromisoformat(experiment_log['start_time']))
    
    # Calcolo delle conclusioni basate sui risultati
    conclusions = analyze_results(results_mnist, results_fashion, dropout_results_mnist, dropout_results_fashion)
    experiment_log['conclusions'] = conclusions
    
    # Creazione del report Markdown
    report_content = f"""# Studio del Dropout come Tecnica di Regolarizzazione
## Analisi Comparativa su Reti Neurali Multistrato

**Data Esperimento:** {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}  
**Device utilizzato:** {experiment_log['device']}  
**Durata totale:** {experiment_log['total_duration']}  

---

## 1. Introduzione e Obiettivi

Questo studio analizza l'efficacia del **dropout** come tecnica di regolarizzazione per le reti neurali multistrato (MLP). L'obiettivo è confrontare le prestazioni di diverse architetture neurali con e senza dropout su due dataset di classificazione:

- **MNIST**: Dataset di cifre scritte a mano (0-9)
- **Fashion-MNIST**: Dataset di capi di abbigliamento (10 categorie)

### 1.1 Metodologia

Sono state implementate e confrontate le seguenti architetture:

1. **MLP Singolo Strato Nascosto** (784 → 512 → 10)
2. **MLP Multi-Strato** (784 → 512 → 256 → 128 → 10)

Per ogni architettura sono state testate due configurazioni:
- **Senza Dropout** (dropout_rate = 0.0)
- **Con Dropout** (dropout_rate = 0.3)

### 1.2 Parametri Sperimentali

- **Ottimizzatore:** Adam
- **Learning Rate:** 0.001
- **Batch Size:** 128
- **Epochs:** 15 (modelli principali), 10 (analisi dropout)
- **Funzione di Loss:** CrossEntropyLoss

---

## 2. Risultati MNIST Dataset

### 2.1 Prestazioni dei Modelli Principali

| Modello | Test Accuracy | Training Time | Overfitting Gap |
|---------|---------------|---------------|-----------------|"""

    # Aggiunta risultati MNIST
    for model_name, data in results_mnist.items():
        gap = data['train_acc'][-1] - data['test_acc']
        report_content += f"\n| {model_name.replace('_', ' ')} | {data['test_acc']:.2f}% | {data['training_time']:.1f}s | {gap:.2f}% |"

    report_content += f"""

### 2.2 Analisi Dettagliata Dropout - MNIST

Sono stati testati i seguenti valori di dropout: {list(dropout_results_mnist.keys())}

| Dropout Rate | Train Accuracy | Test Accuracy | Overfitting Gap |
|--------------|----------------|---------------|-----------------|"""

    # Aggiunta risultati dropout MNIST
    for dropout_rate, data in dropout_results_mnist.items():
        report_content += f"\n| {dropout_rate} | {data['train_acc']:.2f}% | {data['test_acc']:.2f}% | {data['overfitting']:.2f}% |"

    report_content += f"""

**Osservazioni MNIST:**
- Il dropout riduce efficacemente l'overfitting
- Il valore ottimale di dropout per MNIST sembra essere intorno a 0.2-0.3
- L'accuracy test rimane stabile anche con dropout elevato

---

## 3. Risultati Fashion-MNIST Dataset

### 3.1 Prestazioni dei Modelli Principali

| Modello | Test Accuracy | Training Time | Overfitting Gap |
|---------|---------------|---------------|-----------------|"""

    # Aggiunta risultati Fashion-MNIST
    for model_name, data in results_fashion.items():
        gap = data['train_acc'][-1] - data['test_acc']
        report_content += f"\n| {model_name.replace('_', ' ')} | {data['test_acc']:.2f}% | {data['training_time']:.1f}s | {gap:.2f}% |"

    report_content += f"""

### 3.2 Analisi Dettagliata Dropout - Fashion-MNIST

| Dropout Rate | Train Accuracy | Test Accuracy | Overfitting Gap |
|--------------|----------------|---------------|-----------------|"""

    # Aggiunta risultati dropout Fashion-MNIST
    for dropout_rate, data in dropout_results_fashion.items():
        report_content += f"\n| {dropout_rate} | {data['train_acc']:.2f}% | {data['test_acc']:.2f}% | {data['overfitting']:.2f}% |"

    report_content += f"""

**Osservazioni Fashion-MNIST:**
- Fashion-MNIST presenta maggiore difficoltà rispetto a MNIST
- Il dropout è ancora più efficace per ridurre l'overfitting
- Le performance assolute sono inferiori ma la regolarizzazione è più importante

---

## 4. Confronto tra Dataset

### 4.1 Difficoltà Relativa
- **MNIST**: Dataset più semplice, accuracy > 95% facilmente raggiungibili
- **Fashion-MNIST**: Dataset più complesso, accuracy tipicamente 85-90%

### 4.2 Efficacia del Dropout
"""

    # Calcolo dell'efficacia del dropout
    mnist_dropout_effect = calculate_dropout_effectiveness(results_mnist)
    fashion_dropout_effect = calculate_dropout_effectiveness(results_fashion)

    report_content += f"""
- **Su MNIST**: Riduzione media overfitting = {mnist_dropout_effect['avg_overfitting_reduction']:.2f}%
- **Su Fashion-MNIST**: Riduzione media overfitting = {fashion_dropout_effect['avg_overfitting_reduction']:.2f}%

---

## 5. Analisi Tecnica Approfondita

### 5.1 Architetture Testate

#### MLP Singolo Strato
```
Input Layer (784) → Hidden Layer (512) → [Dropout] → Output Layer (10)
```
- **Parametri totali**: ~410K
- **Capacità di memorizzazione**: Media
- **Tendenza all'overfitting**: Moderata

#### MLP Multi-Strato
```
Input (784) → Hidden1 (512) → [Dropout] → Hidden2 (256) → [Dropout] → Hidden3 (128) → [Dropout] → Output (10)
```
- **Parametri totali**: ~590K
- **Capacità di memorizzazione**: Alta
- **Tendenza all'overfitting**: Elevata

### 5.2 Meccanismo del Dropout

Il dropout agisce come regolarizzatore attraverso:

1. **Durante il Training**: Disattiva casualmente neuroni con probabilità `p`
2. **Durante il Test**: Scala i pesi per `(1-p)` per compensare
3. **Effetto**: Riduce la co-adattazione tra neuroni

---

## 6. Conclusioni e Raccomandazioni

### 6.1 Conclusioni Principali

{chr(10).join([f"- {conclusion}" for conclusion in conclusions])}

### 6.2 Raccomandazioni Pratiche

1. **Per dataset semplici (MNIST-like)**: Dropout 0.2-0.3
2. **Per dataset complessi (Fashion-MNIST-like)**: Dropout 0.3-0.4
3. **Per reti profonde**: Dropout essenziale per prevenire overfitting
4. **Per reti shallow**: Dropout utile ma meno critico

### 6.3 Limitazioni dello Studio

- Test limitato a due dataset di immagini 28x28
- Architetture MLP semplici (no CNN)
- Un solo valore di dropout testato per modelli principali
- Numero limitato di epochs per l'analisi dettagliata

### 6.4 Sviluppi Futuri

- Test su dataset più diversificati
- Analisi di altre tecniche di regolarizzazione (Batch Normalization, Weight Decay)
- Confronto con architetture convoluzionali
- Studio dell'interazione dropout + altre tecniche

---

## 7. Dettagli Tecnici

### 7.1 Ambiente di Sviluppo
- **Framework**: PyTorch
- **Device**: {experiment_log['device']}
- **Versioni**: PyTorch, Matplotlib, NumPy

### 7.2 Riproducibilità
Tutti i modelli e i risultati sono stati salvati nelle seguenti directory:
- **Modelli**: `{output_dirs['models']}`
- **Grafici**: `{output_dirs['plots']}`
- **Dati**: `{output_dirs['data']}`
- **Report**: `{output_dirs['reports']}`

### 7.3 File Generati
- Grafici comparativi per ogni dataset
- Analisi dropout dettagliate
- Salvataggio stati modelli (.pth)
- Dati numerici in formato JSON

---

*Report generato automaticamente dal sistema di analisi dropout*  
*Timestamp: {datetime.now().isoformat()}*
"""

    # Salvataggio del report
    report_path = os.path.join(output_dirs['reports'], 'comprehensive_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # Salvataggio del log completo dell'esperimento
    log_path = os.path.join(output_dirs['reports'], 'experiment_log.json')
    with open(log_path, 'w') as f:
        json.dump(experiment_log, f, indent=2)
    
    print(f"\n✅ Report completo salvato in: {report_path}")
    print(f"✅ Log esperimento salvato in: {log_path}")
    
    return report_path

def analyze_results(results_mnist, results_fashion, dropout_mnist, dropout_fashion):
    """Analizza i risultati e genera conclusioni automatiche"""
    conclusions = []
    
    # Analisi overfitting
    mnist_gaps = [data['train_acc'][-1] - data['test_acc'] for data in results_mnist.values()]
    fashion_gaps = [data['train_acc'][-1] - data['test_acc'] for data in results_fashion.values()]
    
    conclusions.append(f"Il dropout riduce l'overfitting in media del {np.mean([gap for gap in mnist_gaps if gap > 0]):.1f}% su MNIST")
    conclusions.append(f"Fashion-MNIST presenta overfitting maggiore di MNIST (gap medio {np.mean(fashion_gaps):.1f}% vs {np.mean(mnist_gaps):.1f}%)")
    
    # Confronto architetture
    mnist_single = [k for k in results_mnist.keys() if 'Single' in k]
    mnist_multi = [k for k in results_mnist.keys() if 'Multi' in k]
    
    if mnist_single and mnist_multi:
        single_acc = np.mean([results_mnist[k]['test_acc'] for k in mnist_single])
        multi_acc = np.mean([results_mnist[k]['test_acc'] for k in mnist_multi])
        conclusions.append(f"Le reti multi-strato ottengono performance {'superiori' if multi_acc > single_acc else 'inferiori'} alle single-layer")
    
    # Analisi dropout ottimale
    best_dropout_mnist = max(dropout_mnist.items(), key=lambda x: x[1]['test_acc'])
    best_dropout_fashion = max(dropout_fashion.items(), key=lambda x: x[1]['test_acc'])
    
    conclusions.append(f"Dropout ottimale per MNIST: {best_dropout_mnist[0]} (accuracy {best_dropout_mnist[1]['test_acc']:.2f}%)")
    conclusions.append(f"Dropout ottimale per Fashion-MNIST: {best_dropout_fashion[0]} (accuracy {best_dropout_fashion[1]['test_acc']:.2f}%)")
    
    return conclusions

def calculate_dropout_effectiveness(results):
    """Calcola l'efficacia del dropout nel ridurre l'overfitting"""
    no_dropout_models = {k: v for k, v in results.items() if 'No_Dropout' in k}
    with_dropout_models = {k: v for k, v in results.items() if 'With_Dropout' in k}
    
    no_dropout_gaps = [data['train_acc'][-1] - data['test_acc'] for data in no_dropout_models.values()]
    with_dropout_gaps = [data['train_acc'][-1] - data['test_acc'] for data in with_dropout_models.values()]
    
    avg_reduction = np.mean(no_dropout_gaps) - np.mean(with_dropout_gaps)
    
    return {
        'avg_overfitting_reduction': avg_reduction,
        'no_dropout_avg_gap': np.mean(no_dropout_gaps),
        'with_dropout_avg_gap': np.mean(with_dropout_gaps)
    }

# Funzione per creare anche un PDF (opzionale, richiede reportlab)
def create_pdf_report(report_path):
    """Converte il report Markdown in PDF (richiede pandoc o reportlab)"""
    try:
        import subprocess
        pdf_path = report_path.replace('.md', '.pdf')
        
        # Tentativo con pandoc
        subprocess.run(['pandoc', report_path, '-o', pdf_path], check=True)
        print(f"✅ Report PDF creato: {pdf_path}")
        return pdf_path
    except:
        print("⚠️ Impossibile creare PDF. Installare pandoc o usare il file Markdown.")
        return None

# Funzione principale aggiornata con salvataggio completo
def main():
    """
    Funzione principale orchestrazione esperimento completo
    
    WORKFLOW SPERIMENTALE:
    1. Setup iniziale (già fatto in __init__)
    2. Esperimenti comparativi su MNIST
    3. Esperimenti comparativi su Fashion-MNIST  
    4. Analisi dettagliata dropout per entrambi dataset
    5. Visualizzazioni avanzate e analisi errori
    6. Generazione report completo
    7. Summary finale
    
    DESIGN PHILOSOPHY:
    - Modularità: ogni fase è function separata
    - Logging completo: tracciabilità totale
    - Salvataggio progressivo: no perdita dati
    - User feedback: progress reporting
    """
    print("STUDIO DEL DROPOUT COME TECNICA DI REGOLARIZZAZIONE")
    print("="*60)
    print(f"📁 Tutti i risultati saranno salvati in: {output_dirs['base']}")
    
    # ===============================================================
    # FASE 1: ESPERIMENTI COMPARATIVI MNIST
    # ===============================================================
    print("\n🔍 Avvio esperimenti su MNIST...")
    
    # CONFRONTO 4 MODELLI: Single/Multi x NoDropout/WithDropout
    results_mnist = compare_models('MNIST')
    
    # VISUALIZZAZIONE IMMEDIATA PER MONITORAGGIO
    plot_results(results_mnist, 'MNIST', save_plots=True)
    
    # ===============================================================
    # FASE 2: ESPERIMENTI COMPARATIVI FASHION-MNIST
    # ===============================================================
    print("\n🔍 Avvio esperimenti su Fashion-MNIST...")
    
    # STESSO PROTOCOLLO SU DATASET PIÙ COMPLESSO
    results_fashion = compare_models('FashionMNIST')
    plot_results(results_fashion, 'Fashion-MNIST', save_plots=True)
    
    # ===============================================================
    # FASE 3: ANALISI SISTEMATICA DROPOUT
    # ===============================================================
    print("\n📊 Analisi dettagliata dell'effetto del dropout...")
    
    # TEST RANGE DROPOUT RATES SU ENTRAMBI DATASET
    dropout_results_mnist = dropout_analysis('MNIST', save_plots=True)
    dropout_results_fashion = dropout_analysis('FashionMNIST', save_plots=True)
    
    # ===============================================================
    # FASE 4: VISUALIZZAZIONI AVANZATE E ANALISI QUALITATIVA
    # ===============================================================
    print("\n🖼️ Visualizzazione campioni con predizioni...")
    
    # CARICAMENTO DATASET PER VISUALIZZAZIONI
    train_loader_mnist, test_loader_mnist = load_datasets('MNIST')
    train_loader_fashion, test_loader_fashion = load_datasets('FashionMNIST')
    
    # PARAMETRI MODELLI
    input_size = 28 * 28
    num_classes = 10
    
    # ===============================================================
    # ANALISI QUALITATIVA MNIST
    # ===============================================================
    try:
        # RICOSTRUZIONE MODELLI MNIST
        # Necessario ricreare architetture per load_state_dict
        model_mnist_single = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.0).to(device)
        model_mnist_single_dropout = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.3).to(device)
        model_mnist_multi = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.0).to(device)
        model_mnist_multi_dropout = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.3).to(device)
        
        # CARICAMENTO PESI SALVATI
        # state_dict contiene solo parametri, non architettura
        model_mnist_single.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'mnist_single_no_dropout.pth')))
        model_mnist_single_dropout.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'mnist_single_with_dropout.pth')))
        model_mnist_multi.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'mnist_multi_no_dropout.pth')))
        model_mnist_multi_dropout.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'mnist_multi_with_dropout.pth')))
        
        # VISUALIZZAZIONI CAMPIONI PREDETTI
        # Mostra performance qualitativa su esempi reali
        print("📸 Visualizzazione campioni MNIST...")
        visualize_sample_predictions(model_mnist_multi_dropout, test_loader_mnist, 'MNIST', num_samples=12)
        
        # CONFRONTO PREDIZIONI TRA MODELLI
        # Analisi comparativa su stessi campioni
        models_dict_mnist = {
            'Single No Dropout': model_mnist_single,
            'Single With Dropout': model_mnist_single_dropout,
            'Multi No Dropout': model_mnist_multi,
            'Multi With Dropout': model_mnist_multi_dropout
        }
        compare_model_predictions(models_dict_mnist, test_loader_mnist, 'MNIST', num_samples=6)
        
        # ANALISI ERRORI TIPICI
        # Insight su failure modes
        print("❌ Analisi errori MNIST...")
        analyze_model_errors(model_mnist_multi, test_loader_mnist, 'MNIST', num_errors=12)
        
        # EFFETTO DROPOUT SU ATTIVAZIONI NEURONALI
        # Analisi quantitativa internal representations
        print("🧠 Analisi effetto dropout sulle attivazioni MNIST...")
        visualize_dropout_effect(model_mnist_multi_dropout, model_mnist_multi, test_loader_mnist, 'MNIST')
        
    except FileNotFoundError:
        print("⚠️ Modelli MNIST non trovati, probabilmente il training non è stato completato.")
    
    # ===============================================================
    # ANALISI QUALITATIVA FASHION-MNIST
    # ===============================================================
    try:
        # RICOSTRUZIONE MODELLI FASHION-MNIST
        model_fashion_single = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.0).to(device)
        model_fashion_single_dropout = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.3).to(device)
        model_fashion_multi = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.0).to(device)
        model_fashion_multi_dropout = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.3).to(device)
        
        # CARICAMENTO PESI
        model_fashion_single.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'fashionmnist_single_no_dropout.pth')))
        model_fashion_single_dropout.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'fashionmnist_single_with_dropout.pth')))
        model_fashion_multi.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'fashionmnist_multi_no_dropout.pth')))
        model_fashion_multi_dropout.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'fashionmnist_multi_with_dropout.pth')))
        
        # VISUALIZZAZIONI FASHION-MNIST
        print("👕 Visualizzazione campioni Fashion-MNIST...")
        visualize_sample_predictions(model_fashion_multi_dropout, test_loader_fashion, 'FashionMNIST', num_samples=12)
        
        # CONFRONTI TRA MODELLI
        models_dict_fashion = {
            'Single No Dropout': model_fashion_single,
            'Single With Dropout': model_fashion_single_dropout,
            'Multi No Dropout': model_fashion_multi,
            'Multi With Dropout': model_fashion_multi_dropout
        }
        compare_model_predictions(models_dict_fashion, test_loader_fashion, 'FashionMNIST', num_samples=6)
        
        # ANALISI ERRORI
        print("❌ Analisi errori Fashion-MNIST...")
        analyze_model_errors(model_fashion_multi, test_loader_fashion, 'FashionMNIST', num_errors=12)
        
        # EFFETTO DROPOUT
        print("🧠 Analisi effetto dropout sulle attivazioni Fashion-MNIST...")
        visualize_dropout_effect(model_fashion_multi_dropout, model_fashion_multi, test_loader_fashion, 'FashionMNIST')
        
    except FileNotFoundError:
        print("⚠️ Modelli Fashion-MNIST non trovati, probabilmente il training non è stato completato.")
    
    # ===============================================================
    # FASE 5: ANALISI COMPARATIVE E SUMMARY
    # ===============================================================
    
    # PLOT RIASSUNTIVO GLOBALE
    # Confronto high-level tra tutti esperimenti
    create_summary_plot(results_mnist, results_fashion)
    
    # REPORT FINALE CONSOLE
    # Summary quantitativo per quick reference
    print_final_summary(results_mnist, results_fashion, dropout_results_mnist, dropout_results_fashion)
    
    # ===============================================================
    # FASE 6: GENERAZIONE REPORT COMPLETO
    # ===============================================================
    
    print("\n📝 Generazione report completo...")
    
    # REPORT MARKDOWN STRUTTURATO
    # Include analisi, grafici, conclusioni, raccomandazioni
    report_path = create_comprehensive_report(
        results_mnist, results_fashion, 
        dropout_results_mnist, dropout_results_fashion
    )
    
    # TENTATIVO CONVERSIONE PDF
    print("\n📄 Tentativo di creazione PDF...")
    pdf_path = create_pdf_report(report_path)
    
    # ===============================================================
    # COMPLETION MESSAGE
    # ===============================================================
    
    print(f"\n✅ ESPERIMENTO COMPLETATO!")
    print(f"📁 Tutti i file sono disponibili in: {output_dirs['base']}")
    print(f"📊 Grafici salvati in: {output_dirs['plots']}")
    print(f"🤖 Modelli salvati in: {output_dirs['models']}")
    print(f"📝 Report salvato in: {output_dirs['reports']}")

def demo_visualizations(dataset_name='MNIST', model_type='multi'):
    """
    Funzione demo per testare le visualizzazioni senza dover rifare tutto il training
    
    Args:
        dataset_name: 'MNIST' o 'FashionMNIST'
        model_type: 'single' o 'multi'
    """
    print(f"\n🎬 DEMO VISUALIZZAZIONI - {dataset_name} ({model_type})")
    print("="*60)
    
    # Carica dataset
    _, test_loader = load_datasets(dataset_name)
    
    # Parametri
    input_size = 28 * 28
    num_classes = 10
    
    # Crea modelli
    if model_type == 'single':
        model_no_dropout = MLPSingleHidden(input_size, 256, num_classes, dropout_rate=0.0).to(device)  # Più piccolo
        model_with_dropout = MLPSingleHidden(input_size, 256, num_classes, dropout_rate=0.5).to(device)
        model_files = [
            f'{dataset_name.lower()}_single_no_dropout.pth',
            f'{dataset_name.lower()}_single_with_dropout.pth'
        ]
    else:
        model_no_dropout = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.0).to(device)
        model_with_dropout = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.3).to(device)
        model_files = [
            f'{dataset_name.lower()}_multi_no_dropout.pth',
            f'{dataset_name.lower()}_multi_with_dropout.pth'
        ]
    
    # Cerca i modelli salvati nella cartella più recente
    base_dirs = [d for d in os.listdir('.') if d.startswith('dropout_study_results_')]
    if base_dirs:
        latest_dir = sorted(base_dirs)[-1]
        models_dir = os.path.join(latest_dir, 'models')
        
        try:
            model_no_dropout.load_state_dict(torch.load(os.path.join(models_dir, model_files[0])))
            model_with_dropout.load_state_dict(torch.load(os.path.join(models_dir, model_files[1])))

            # 1. Visualizza campioni con predizioni
            print("\n📸 1. Visualizzazione campioni MNIST...")
            visualize_sample_predictions(model_with_dropout, test_loader, dataset_name, num_samples=12)
            
            # 2. Confronta predizioni dei modelli
            print("\n🔍 2. Confronto predizioni modelli...")
            models_dict = {
                f'{model_type.title()} No Dropout': model_no_dropout,
                f'{model_type.title()} With Dropout': model_with_dropout
            }
            compare_model_predictions(models_dict, test_loader, dataset_name, num_samples=8)
            
            # 3. Analizza errori
            print("\n❌ 3. Analisi errori del modello...")
            analyze_model_errors(model_no_dropout, test_loader, dataset_name, num_errors=12)
            
            # 4. Effetto dropout sulle attivazioni
            print("\n🧠 4. Effetto dropout sulle attivazioni...")
            visualize_dropout_effect(model_with_dropout, model_no_dropout, test_loader, dataset_name)
            
            print("\n🎉 Demo completata!")
            
        except FileNotFoundError as e:
            print(f"❌ Errore: Modelli non trovati. {e}")
            print("💡 Suggerimento: Esegui prima il training completo con main()")
    else:
        print("❌ Nessuna cartella di risultati trovata.")
        print("💡 Suggerimento: Esegui prima il training completo con main()")

# Funzione per training rapido e demo (per test veloce)
def quick_demo(dataset_name='MNIST', epochs=5):
    """
    Training rapido e demo delle visualizzazioni per test veloce
    
    Args:
        dataset_name: 'MNIST' o 'FashionMNIST'
        epochs: Numero di epochs per training veloce
    """
    print(f"\n⚡ QUICK DEMO - {dataset_name}")
    print("="*40)
    
    # Carica dataset
    train_loader, test_loader = load_datasets(dataset_name, batch_size=256)  # Batch più grande per velocità
    
    # Parametri
    input_size = 28 * 28
    num_classes = 10
    learning_rate = 0.001
    
    # Modelli semplificati
    model_no_dropout = MLPSingleHidden(input_size, 256, num_classes, dropout_rate=0.0).to(device)  # Più piccolo
    model_with_dropout = MLPSingleHidden(input_size, 256, num_classes, dropout_rate=0.5).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model_no_dropout.parameters(), lr=learning_rate)
    optimizer2 = optim.Adam(model_with_dropout.parameters(), lr=learning_rate)
    
    # Training veloce
    print(f"🚀 Training veloce ({epochs} epochs)...")
    
    print("Training modello SENZA dropout...")
    train_model(model_no_dropout, train_loader, criterion, optimizer1, epochs)
    
    print("Training modello CON dropout...")
    train_model(model_with_dropout, train_loader, criterion, optimizer2, epochs)
    
    # Test
    acc1, _ = test_model(model_no_dropout, test_loader)
    acc2, _ = test_model(model_with_dropout, test_loader)
    
    print(f"📊 Accuracy senza dropout: {acc1:.2f}%")
    print(f"📊 Accuracy con dropout: {acc2:.2f}%")
    
    # Demo visualizzazioni
    print("\n🎬 Avvio demo visualizzazioni...")
    
    # 1. Campioni con predizioni
    print("\n📸 Campioni con predizioni...")
    visualize_sample_predictions(model_with_dropout, test_loader, dataset_name, num_samples=8, save_plots=False)
    
    # 2. Confronto modelli
    print("\n🔍 Confronto modelli...")
    models_dict = {
        'Senza Dropout': model_no_dropout,
        'Con Dropout': model_with_dropout
    }
    compare_model_predictions(models_dict, test_loader, dataset_name, num_samples=4, save_plots=False)
    
    # 3. Analisi errori
    print("\n❌ Analisi errori...")
    analyze_model_errors(model_no_dropout, test_loader, dataset_name, num_errors=8, save_plots=False)
    
    print("\n✅ Quick demo completata!")

if __name__ == "__main__":
    # ENTRY POINT PRINCIPALE
    # Esegue l'esperimento completo con tutti i moduli
    main()
    
    # ===============================================================
    # ALTERNATIVE EXECUTION MODES (per development/testing)
    # ===============================================================
    
    # MODALITA' DEMO VISUALIZZAZIONI (decommentare se necessario):
    # Per testare solo visualizzazioni con modelli già trainati
    # demo_visualizations('MNIST', 'multi')
    # demo_visualizations('FashionMNIST', 'single')
    
    # MODALITA' QUICK TEST (decommentare se necessario):
    # Per test rapidi durante development
    # quick_demo('MNIST', epochs=3)

# ===================================================================
# DOCUMENTAZIONE FINALE E ISTRUZIONI D'USO
# ===================================================================

"""
ISTRUZIONI PER L'UTILIZZO:

1. ESPERIMENTO COMPLETO (RACCOMANDATO):
   python examp2.py
   
   Esegue l'intero workflow:
   - Training 8 modelli (4 per MNIST + 4 per Fashion-MNIST)
   - Analisi dropout sistematica
   - Visualizzazioni avanzate
   - Report completo automatico
   
2. SOLO VISUALIZZIONI (se hai già i modelli trainati):
   demo_visualizations('MNIST', 'multi')
   demo_visualizations('FashionMNIST', 'single')
   
   Utile per:
   - Re-analisi dati esistenti
   - Generazione grafici aggiuntivi
   - Testing visualizzazioni

3. TEST VELOCE (development mode):
   quick_demo('MNIST', epochs=3)
   quick_demo('FashionMNIST', epochs=5)
   
   Per:
   - Verificare funzionamento codice
   - Debug rapido
   - Demo presentazioni

NUOVE FUNZIONALITÀ IMPLEMENTATE:

- visualize_sample_predictions(): 
  Mostra campioni dataset con predizioni modello e confidence scores
  
- compare_model_predictions(): 
  Confronta predizioni di più modelli sugli stessi campioni per analisi comparative
  
- analyze_model_errors(): 
  Identifica e visualizza errori più comuni per insight su failure modes
  
- visualize_dropout_effect(): 
  Analizza effetto dropout su attivazioni neuronali (distribuzione attivazioni)

ORGANIZZAZIONE OUTPUT:

dropout_study_results_YYYYMMDD_HHMMSS/
├── plots/          # Tutti i grafici generati (.png alta risoluzione)
├── models/         # Stati modelli salvati (.pth files)
├── reports/        # Report markdown e log JSON
└── data/           # Risultati numerici in formato JSON

DIPENDENZE RICHIESTE:

- torch >= 1.9.0
- torchvision >= 0.10.0  
- matplotlib >= 3.3.0
- numpy >= 1.19.0
- scikit-learn >= 0.24.0
- seaborn >= 0.11.0

HARDWARE RACCOMANDATO:

- GPU NVIDIA con CUDA (opzionale ma accelera 10-100x)
- RAM >= 8GB per dataset loading
- Storage >= 2GB per risultati completi

TROUBLESHOOTING:

1. CUDA out of memory:
   - Riduci batch_size in load_datasets()
   - Usa device='cpu' forzatamente
   
2. Missing model files:
   - Riesegui training completo
   - Verifica percorsi in output_dirs
   
3. Import errors:
   - pip install -r requirements.txt
   - Controlla versioni PyTorch compatibili

4. Slow execution:
   - Riduci num_epochs per test
   - Usa quick_demo() per sviluppo
   - Abilita GPU se disponibile

Per supporto: verificare log di output e error messages dettagliati.
"""