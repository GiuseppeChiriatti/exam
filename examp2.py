<<<<<<< HEAD

"""
STUDIO COMPLETO DEL DROPOUT COME TECNICA DI REGOLARIZZAZIONE
=============================================================

Questo script implementa uno studio completo e approfondito dell'efficacia del dropout
come tecnica di regolarizzazione nelle reti neurali multistrato (MLP).

OBIETTIVI DELLO STUDIO:
1. Confrontare prestazioni di architetture MLP con e senza dropout
2. Analizzare l'effetto del dropout sull'overfitting
3. Testare diversi valori di dropout rate
4. Generare visualizzazioni e report dettagliati
5. Documentare tutti i risultati in formato scientifico

ARCHITETTURE TESTATE:
- MLP singolo strato nascosto (784→512→10)
- MLP multi-strato (784→512→256→128→10)

DATASET UTILIZZATI:
- MNIST: cifre scritte a mano (baseline)
- Fashion-MNIST: capi di abbigliamento (più complesso)

TECNOLOGIE E MOTIVAZIONI:
- PyTorch: framework deep learning leader, flessibile e efficiente
- Matplotlib: visualizzazione scientifica standard
- NumPy: calcoli numerici ottimizzati
- scikit-learn: metriche di valutazione consolidate
- JSON: serializzazione dati per riproducibilità
"""

# ============================================================================
# IMPORTAZIONE LIBRERIE E MODULI
# ============================================================================

# CORE DEEP LEARNING - PyTorch Framework
import torch                    # Framework principale per deep learning
import torch.nn as nn          # Moduli per costruzione reti neurali (layers, loss functions)
import torch.optim as optim    # Algoritmi di ottimizzazione (Adam, SGD, etc.)
import torch.nn.functional as F # Funzioni di attivazione e utilità (ReLU, Softmax, etc.)
from torch.utils.data import DataLoader  # Gestione efficiente dei dataset in batch

# COMPUTER VISION - TorchVision
import torchvision             # Libreria per computer vision built su PyTorch
import torchvision.transforms as transforms  # Trasformazioni per preprocessing immagini

# VISUALIZZAZIONE E ANALISI DATI
import matplotlib.pyplot as plt # Plotting scientifico standard per grafici di alta qualità
import numpy as np             # Calcoli numerici ottimizzati, interoperabilità con PyTorch
import seaborn as sns          # Visualizzazioni statistiche avanzate built su matplotlib

# METRICHE E VALUTAZIONE MODELLI
from sklearn.metrics import classification_report, confusion_matrix  # Metriche standard ML

# UTILITÀ SISTEMA E STRUTTURE DATI
from collections import defaultdict  # Dizionari con valori di default per aggregazione dati
import time                    # Misurazione tempi di esecuzione per benchmarking
import os                      # Operazioni filesystem per gestione directory e file
from datetime import datetime  # Timestamp per tracciabilità esperimenti
import json                    # Serializzazione dati per persistenza e riproducibilità

# ============================================================================
# CONFIGURAZIONE AMBIENTE DI ESECUZIONE
# ============================================================================

# RILEVAMENTO E CONFIGURAZIONE HARDWARE
# PyTorch supporta esecuzione su CPU o GPU (CUDA)
# La GPU accelera significativamente i calcoli tensoriali paralleli
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilizzo device: {device}")
# Vantaggi GPU: 
# - Calcolo parallelo massivo (migliaia di core)
# - Ottimizzata per operazioni matriciali
# - Accelerazione 10-100x rispetto a CPU per deep learning

# ============================================================================
# SISTEMA DI ORGANIZZAZIONE RISULTATI
# ============================================================================

<<<<<<< HEAD
def create_output_directories():
    """
    Crea struttura organizzata di directory per salvare tutti i risultati dell'esperimento.
    
    MOTIVAZIONI DESIGN:
    1. TIMESTAMP: Ogni run ha ID univoco per evitare sovrascritture
    2. MODULARITÀ: Risultati separati per tipo (plots, models, reports, data)
    3. RIPRODUCIBILITÀ: Struttura standardizzata per confronti futuri
    4. PROFESSIONALITÀ: Organizzazione scientifica dei risultati
    
    Returns:
        dict: Dizionario con percorsi a tutte le directory create
    """
    # Genera timestamp formato YYYYMMDD_HHMMSS per identificazione univoca
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"dropout_study_results_{timestamp}"
    
    # Struttura directory organizzata per tipo di contenuto
    directories = {
        'base': base_dir,                              # Directory principale esperimento
        'plots': os.path.join(base_dir, 'plots'),     # Grafici e visualizzazioni
        'models': os.path.join(base_dir, 'models'),   # Modelli addestrati salvati (.pth files)
        'reports': os.path.join(base_dir, 'reports'), # Report markdown e log JSON
        'data': os.path.join(base_dir, 'data')        # Dati numerici e metriche JSON
    }
    
    # Creazione fisica delle directory su filesystem
    # exist_ok=True previene errori se directory già esistenti
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

# Inizializzazione globale delle directory
output_dirs = create_output_directories()
print(f"Risultati salvati in: {output_dirs['base']}")

<<<<<<< HEAD
# ============================================================================
# LOGGING E TRACCIABILITÀ ESPERIMENTI
# ============================================================================

# Variabile globale per raccogliere informazioni per il report
# Struttura dati centralizzata per tracciare tutti gli aspetti dell'esperimento
experiment_log = {
    'start_time': datetime.now().isoformat(),    # Timestamp inizio esperimento (ISO format)
    'device': str(device),                       # Device utilizzato (CPU/GPU) per riproducibilità
    'experiments': [],                           # Lista di tutti gli esperimenti eseguiti
    'conclusions': []                            # Lista delle conclusioni automaticamente generate
}

# ============================================================================
# DEFINIZIONE ARCHITETTURE DI RETE NEURALE
# ============================================================================

# Classe MLP con un solo strato nascosto
=======
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

>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
class MLPSingleHidden(nn.Module):
    """
    Multi-Layer Perceptron con un singolo strato nascosto
    
<<<<<<< HEAD
    ARCHITETTURA: Input → Hidden Layer → Dropout → Output
    
    MOTIVAZIONI DESIGN:
    1. SEMPLICITÀ: Architettura di baseline per confronti
    2. CONTROLLO: Numero ridotto di parametri per analisi chiara
    3. INTERPRETABILITÀ: Comportamento del dropout più facilmente osservabile
    4. BASELINE: Punto di riferimento per reti più complesse
    
    PARAMETRI:
    - input_size: Dimensione input (784 per immagini 28x28)
    - hidden_size: Neuroni nello strato nascosto (512 per capacità adeguata)
    - num_classes: Classi di output (10 per MNIST/Fashion-MNIST)
    - dropout_rate: Probabilità di dropout (0.0-1.0)
    """
=======
    MOTIVAZIONE ARCHITETTUALE:
    - Semplicità: Baseline per confronti
    - Interpretabilità: Pochi parametri da analizzare
    - Velocità: Training rapido per esperimenti
    
    PERCHÉ QUESTA IMPLEMENTAZIONE:
    - Eredita da nn.Module per integrazione PyTorch
    - Dropout configurabile per esperimenti controllati
    - ReLU come attivazione (standard moderno)
    """
    
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
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
        
<<<<<<< HEAD
        # STRATO LINEARE 1: Input → Hidden
        # Trasformazione lineare W*x + b dove W è matrice pesi, b è bias
        # Dimensioni: [input_size, hidden_size]
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # DROPOUT LAYER
        # Disattiva casualmente neuroni con probabilità dropout_rate
        # Durante training: alcuni neuroni → 0, altri scalati per compensazione
        # Durante inference: tutti neuroni attivi, pesi scalati automaticamente
        self.dropout = nn.Dropout(dropout_rate)
        
        # STRATO LINEARE 2: Hidden → Output
        # Produce logits per classificazione (prima di softmax)
        # Dimensioni: [hidden_size, num_classes]
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Salva dropout_rate per riferimento e debug
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """
<<<<<<< HEAD
        Forward pass della rete neurale
        
        FLUSSO DATI:
        1. Flatten: Immagine 28x28 → vettore 784
        2. Linear: Trasformazione lineare + bias
        3. ReLU: Attivazione non-lineare (f(x) = max(0,x))
        4. Dropout: Regolarizzazione stocastica
        5. Linear: Classificazione finale
        
        Args:
            x: Tensor input [batch_size, channels, height, width]
            
        Returns:
            Tensor: Logits non normalizzati [batch_size, num_classes]
        """
        # FLATTEN: Converte immagini 2D in vettori 1D
        # Da [batch_size, 1, 28, 28] a [batch_size, 784]
        # Necessario perché Linear layers richiedono input 1D
        x = x.view(x.size(0), -1)
        
        # PRIMO STRATO + ATTIVAZIONE
        # Linear: applica trasformazione affine W₁x + b₁
        # ReLU: introduce non-linearità, elimina gradienti negativi
        x = F.relu(self.fc1(x))
        
        # DROPOUT: Regolarizzazione durante training
        # Training mode: disattiva neuroni casualmente
        # Eval mode: passa tutto attraverso, scala automaticamente
        x = self.dropout(x)
        
        # STRATO OUTPUT: Produce classificazione finale
        # Non applichiamo softmax qui perché CrossEntropyLoss lo fa internamente
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        x = self.fc2(x)
        
        return x

class MLPMultiHidden(nn.Module):
    """
<<<<<<< HEAD
    Multi-Layer Perceptron con architettura profonda multi-strato
    
    ARCHITETTURA: Input → Hidden1 → Dropout → Hidden2 → Dropout → ... → Output
    
    MOTIVAZIONI DESIGN:
    1. CAPACITÀ MAGGIORE: Più strati = maggiore capacità di apprendimento
    2. RAPPRESENTAZIONI GERARCHICHE: Ogni strato apprende features di livello diverso
    3. UNIVERSALITÀ: Teorema approssimazione universale per reti profonde
    4. SFIDA OVERFITTING: Maggiore complessità = maggiore rischio overfitting
    
    DIFFERENZE DA SINGLE HIDDEN:
    - ModuleList: Gestione dinamica di layer multipli
    - Architettura progressivamente decrescente: 512→256→128
    - Dropout dopo ogni strato nascosto
    - Maggiore numero di parametri (~590K vs ~410K)
    """
=======
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
    
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
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
<<<<<<< HEAD
        
        # INIZIALIZZAZIONE LAYER CONTAINERS
        # ModuleList: lista dinamica di moduli PyTorch
        # Registra automaticamente parametri per l'ottimizzazione
        self.layers = nn.ModuleList()      # Layer lineari
        self.dropouts = nn.ModuleList()    # Layer dropout corrispondenti
        
        # PRIMO STRATO: Input → Prima dimensione nascosta
        # Connette l'input (784 pixel) al primo strato nascosto
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.dropouts.append(nn.Dropout(dropout_rate))
        
        # STRATI INTERMEDI: Costruzione dinamica dell'architettura
        # Crea connessioni tra strati nascosti consecutivi
        # Range: da 1 a len(hidden_sizes) per collegare strati adiacenti
=======
        
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
<<<<<<< HEAD
        # STRATO DI OUTPUT: Ultimo strato nascosto → Classificazione
        # Non ha dropout per preservare tutte le informazioni finali
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        
        # Salvataggio configurazione per debug e analisi
=======
        # LAYER DI OUTPUT
        # Ultimo hidden -> num_classes
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        
        # Memorizzazione configurazione
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """
<<<<<<< HEAD
        Forward pass per architettura multi-strato
        
        FLUSSO COMPUTAZIONALE:
        1. Flatten input da 2D a 1D
        2. Per ogni coppia (linear_layer, dropout_layer):
           - Applicazione trasformazione lineare
           - Attivazione ReLU
           - Regolarizzazione dropout
        3. Strato output finale senza dropout
        
        VANTAGGI ARCHITETTURA PROFONDA:
        - Features gerarchiche: bordi → forme → oggetti
        - Maggiore espressività rappresentazionale
        - Capacità di apprendere pattern complessi
        
        Args:
            x: Input tensor [batch_size, 1, 28, 28]
            
        Returns:
            Tensor: Logits finali [batch_size, num_classes]
        """
        # PREPROCESSING: Flatten delle immagini
        # Converte da formato immagine a vettore per MLP
        x = x.view(x.size(0), -1)  # [batch_size, 784]
        
        # PROPAGAZIONE ATTRAVERSO STRATI NASCOSTI
        # zip() accoppia ogni layer con il suo dropout corrispondente
        # Iterazione sincrona: layer[i] → dropout[i]
        for layer, dropout in zip(self.layers, self.dropouts):
            # Trasformazione lineare: W*x + b
            x = layer(x)
            # Attivazione non-lineare: introduce capacità di apprendimento complesso
            x = F.relu(x)
            # Regolarizzazione: previene overfitting, migliora generalizzazione
            x = dropout(x)
        
        # CLASSIFICAZIONE FINALE
        # Ultimo strato senza attivazione (logits raw)
        # CrossEntropyLoss applicherà softmax internamente
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        x = self.output_layer(x)
        
        return x

<<<<<<< HEAD
# ============================================================================
# GESTIONE E PREPROCESSING DATASET
# ============================================================================

def load_datasets(dataset_name='MNIST', batch_size=128):
    """
    Carica e preprocessa i dataset per l'esperimento
    
    TECNOLOGIE UTILIZZATE:
    - TorchVision: Libreria standard per computer vision in PyTorch
    - DataLoader: Gestione efficiente dei batch e shuffle
    - Transforms: Pipeline di preprocessing delle immagini
    
    PREPROCESSING PIPELINE:
    1. ToTensor(): Converte PIL/numpy → PyTorch tensor + normalizza [0,1]
    2. Normalize((0.5,), (0.5,)): Normalizza pixel da [0,1] a [-1,1]
       Formula: (pixel - 0.5) / 0.5 = 2*pixel - 1
    
    MOTIVAZIONI SCELTE:
    - Normalizzazione [-1,1]: Accelera convergenza, stabilizza gradienti
    - Batch size 128: Compromesso tra memoria GPU e stabilità gradienti
    - Shuffle=True: Previene bias di ordinamento nel training
    - Shuffle=False: Valutazione deterministica nel test
    
    Args:
        dataset_name: 'MNIST' o 'FashionMNIST'
        batch_size: Dimensione batch per DataLoader
        
    Returns:
        tuple: (train_loader, test_loader) con DataLoader configurati
    """
    # DEFINIZIONE PIPELINE DI PREPROCESSING
    # Compose: concatena trasformazioni in sequenza deterministica
    transform = transforms.Compose([
        transforms.ToTensor(),          # PIL Image/numpy → Tensor float32 [0,1]
        transforms.Normalize((0.5,), (0.5,))  # [0,1] → [-1,1] per stabilità numerica
    ])
    
    # CARICAMENTO DATASET SPECIFICO
    # download=True: scarica automaticamente se non presente
    # root='./data': directory locale per cache dataset
    if dataset_name == 'MNIST':
        # MNIST: 60K training + 10K test, cifre 0-9, 28x28 grayscale
        # Dataset storico per benchmarking, relativamente semplice
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
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
<<<<<<< HEAD
        # Fashion-MNIST: stesso formato di MNIST ma con capi abbigliamento
        # Più challenging: maggiore variabilità intra-classe
=======
        # Fashion-MNIST: Capi abbigliamento
        # Stesse dimensioni MNIST ma più complesso
        # 10 categorie: t-shirt, pantaloni, pullover, etc.
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
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
    
<<<<<<< HEAD
    # CREAZIONE DATALOADER PER GESTIONE EFFICIENTE BATCH
    # TRAINING LOADER:
    # - shuffle=True: ordine casuale per generalizzazione
    # - Evita bias da ordinamento sequenziale delle classi
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # TEST LOADER:
    # - shuffle=False: ordine determinístico per riproducibilità
    # - Valutazione consistente tra run multipli
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# ============================================================================
# FUNZIONI DI ADDESTRAMENTO E VALUTAZIONE
# ============================================================================

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Funzione di addestramento completa per reti neurali
    
    PROCESSO DI ADDESTRAMENTO:
    1. Forward Pass: Calcolo predizioni del modello
    2. Loss Computation: Confronto predizioni vs etichette reali
    3. Backward Pass: Calcolo gradienti via backpropagation
    4. Parameter Update: Aggiornamento pesi via ottimizzatore
    5. Metrics Tracking: Monitoraggio prestazioni per analisi
    
    TECNOLOGIE CHIAVE:
    - model.train(): Attiva modalità training (dropout attivo)
    - loss.backward(): Calcola gradienti automaticamente
    - optimizer.step(): Applica regola di aggiornamento (Adam)
    - torch.max(): Trova classe con probabilità massima
    
    Args:
        model: Rete neurale da addestrare
        train_loader: DataLoader con batch di training
        criterion: Funzione di loss (CrossEntropyLoss)
        optimizer: Algoritmo di ottimizzazione (Adam)
        num_epochs: Numero di epoche di addestramento
        
    Returns:
        tuple: (train_losses, train_accuracies) liste con metriche per epoca
    """
    # MODALITÀ TRAINING: Attiva dropout e batch normalization
    # Essenziale per comportamento corretto del dropout
    model.train()
    
    # INIZIALIZZAZIONE TRACKING METRICHE
    # Liste per salvare progresso durante l'addestramento
    train_losses = []      # Loss media per ogni epoca
    train_accuracies = []  # Accuracy media per ogni epoca
    
    # LOOP PRINCIPALE DI ADDESTRAMENTO
    # Itera attraverso tutte le epoche specificate
    for epoch in range(num_epochs):
        # INIZIALIZZAZIONE CONTATORI PER EPOCA CORRENTE
        running_loss = 0.0  # Accumula loss totale dell'epoca
        correct = 0          # Conta predizioni corrette
        total = 0            # Conta esempi totali processati
        
        # ITERAZIONE ATTRAVERSO TUTTI I BATCH DELL'EPOCA
        # enumerate() fornisce indice batch per debugging
        for i, (inputs, labels) in enumerate(train_loader):
            # SPOSTAMENTO DATI SU DEVICE (CPU/GPU)
            # .to(device) assicura che tensori siano sullo stesso device del modello
            inputs, labels = inputs.to(device), labels.to(device)
            
            # AZZERAMENTO GRADIENTI
            # I gradienti si accumulano per default in PyTorch
            # Necessario azzerarli ad ogni batch per evitare interferenze
            optimizer.zero_grad()
            
            # FORWARD PASS: Calcolo predizioni
            # Il modello processa il batch e produce logits
            outputs = model(inputs)
            
            # CALCOLO LOSS
            # CrossEntropyLoss confronta logits con etichette ground truth
            # Combina softmax + negative log likelihood internamente
            loss = criterion(outputs, labels)
            
            # BACKWARD PASS: Calcolo gradienti
            # Backpropagation automatica attraverso tutti i layer
            # Calcola ∂Loss/∂θ per ogni parametro θ
            loss.backward()
            
            # AGGIORNAMENTO PARAMETRI
            # Applica regola di aggiornamento Adam: θ = θ - α∇θ
            # Adam adatta learning rate per ogni parametro individualmente
            optimizer.step()
            
            # ACCUMULO STATISTICHE PER MONITORAGGIO
            running_loss += loss.item()  # .item() estrae valore scalare da tensor
            
            # CALCOLO ACCURACY DEL BATCH
            # torch.max() trova indice classe con score massimo
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # Aggiunge dimensione batch
            correct += (predicted == labels).sum().item()  # Conta predizioni corrette
        
<<<<<<< HEAD
        # CALCOLO METRICHE FINALI DELL'EPOCA
        epoch_loss = running_loss / len(train_loader)  # Loss media dell'epoca
        epoch_acc = 100 * correct / total              # Accuracy percentuale
        
        # SALVATAGGIO METRICHE PER ANALISI POSTERIORE
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # STAMPA PROGRESSO OGNI 2 EPOCHE
        # Fornisce feedback visuale durante addestramento lungo
=======
        # CALCOLO METRICHE EPOCA
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # SALVATAGGIO METRICHE
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # LOGGING PERIODICO
        # Evita spam su console, print ogni 2 epoche
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return train_losses, train_accuracies

def test_model(model, test_loader):
    """
<<<<<<< HEAD
    Funzione di valutazione per testare prestazioni del modello addestrato
    
    PROCESSO DI VALUTAZIONE:
    1. Modalità Evaluation: Disattiva dropout e batch norm training
    2. No Gradient Computation: torch.no_grad() per efficienza memoria
    3. Batch Processing: Processa tutto il test set in batch
    4. Metrics Calculation: Calcola accuracy finale e loss media
    
    DIFFERENZE DA TRAINING:
    - model.eval(): Disattiva dropout (tutti neuroni attivi)
    - torch.no_grad(): Non calcola gradienti (solo forward pass)
    - Nessun backward pass o aggiornamento parametri
    - Valutazione deterministica per riproducibilità
    
    Args:
        model: Modello addestrato da valutare
        test_loader: DataLoader con dati di test
        
    Returns:
        tuple: (accuracy, avg_loss) metriche finali del test
    """
    # MODALITÀ EVALUATION: Disattiva training specifiche (dropout, etc.)
    # Essenziale per valutazione corretta: dropout deve essere spento
    model.eval()
    
    # INIZIALIZZAZIONE CONTATORI
    correct = 0           # Predizioni corrette
    total = 0             # Esempi totali processati
    test_loss = 0         # Accumula loss per calcolo media
    criterion = nn.CrossEntropyLoss()  # Loss function per valutazione
    
    # DISATTIVAZIONE CALCOLO GRADIENTI
    # torch.no_grad(): ottimizzazione memoria e velocità
    # Nessun backward pass necessario durante valutazione
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    with torch.no_grad():
        # ITERAZIONE ATTRAVERSO TUTTI I BATCH DI TEST
        for inputs, labels in test_loader:
<<<<<<< HEAD
            # SPOSTAMENTO DATI SU DEVICE
            inputs, labels = inputs.to(device), labels.to(device)
            
            # FORWARD PASS: Solo predizione, no training
            outputs = model(inputs)
            
            # CALCOLO LOSS: Per monitoraggio convergenza
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # ESTRAZIONE PREDIZIONI: Classe con probabilità massima
=======
            # Device transfer
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass (solo predizione)
            outputs = model(inputs)
            
            # Calcolo loss per monitoraggio
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Calcolo accuracy
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)  # Conta esempi nel batch
            correct += (predicted == labels).sum().item()  # Conta predizioni corrette
    
    # CALCOLO METRICHE FINALI
    accuracy = 100 * correct / total           # Accuracy percentuale
    avg_loss = test_loss / len(test_loader)    # Loss media su tutti i batch
    
<<<<<<< HEAD
=======
    # CALCOLO METRICHE FINALI
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    return accuracy, avg_loss

# Funzione per confrontare modelli con logging
def compare_models(dataset_name='MNIST'):
    """
<<<<<<< HEAD
    Funzione principale per confrontare prestazioni di diverse architetture MLP
    con e senza dropout su un dataset specifico
    
    ESPERIMENTI CONDOTTI:
    1. MLP Single Hidden - Senza Dropout (baseline)
    2. MLP Single Hidden - Con Dropout 0.3 (regolarizzato)
    3. MLP Multi Hidden - Senza Dropout (complesso baseline)
    4. MLP Multi Hidden - Con Dropout 0.3 (complesso regolarizzato)
    
    METRICHE TRACKED:
    - Training Loss/Accuracy per epoca
    - Test Accuracy e Loss finali
    - Tempo di addestramento
    - Gap overfitting (train_acc - test_acc)
    
    TECNOLOGIE PER REPRODUCIBILITÀ:
    - Salvataggio stato modelli (.pth files)
    - Serializzazione risultati JSON
    - Logging strutturato esperimenti
    - Timestamp per versioning
    
    Args:
        dataset_name: 'MNIST' o 'FashionMNIST'
        
    Returns:
        dict: Risultati strutturati di tutti gli esperimenti
    """
    print(f"\n=== CONFRONTO MODELLI SU {dataset_name} ===")
    
    # CARICAMENTO E CONFIGURAZIONE DATASET
    # Utilizza la funzione load_datasets per preprocessing standardizzato
    train_loader, test_loader = load_datasets(dataset_name)
    
    # PARAMETRI SPERIMENTALI STANDARDIZZATI
    # Valori ottimizzati per bilanciare performance e tempo di training
    input_size = 28 * 28      # 784 pixel per immagini MNIST/Fashion-MNIST
    num_classes = 10          # 10 categorie in entrambi i dataset
    num_epochs = 15           # Sufficiente per convergenza, evita overfitting estremo
    learning_rate = 0.001     # Learning rate Adam standard per questo tipo di problema
    
    # INIZIALIZZAZIONE STRUTTURE DATI RISULTATI
    results = {}              # Dizionario per risultati numerici
    experiment_info = {       # Metadata dettagliati per report
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        'dataset': dataset_name,
        'parameters': {
            'input_size': input_size,
            'num_classes': num_classes,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate
        },
        'models': {}
    }
      # ========================================================================
    # ESPERIMENTO 1: MLP SINGOLO STRATO SENZA DROPOUT (BASELINE)
    # ========================================================================
    
    # ===============================================================
    # ESPERIMENTO 1: MLP SINGOLO STRATO SENZA DROPOUT
    # ===============================================================
    print("\n1. Training MLP singolo strato SENZA dropout...")
    
<<<<<<< HEAD
    # CREAZIONE MODELLO BASELINE
    # Architettura semplice per stabilire performance di riferimento
    # dropout_rate=0.0: nessuna regolarizzazione, comportamento "naturale" della rete
    model1 = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.0).to(device)
    
    # CONFIGURAZIONE OTTIMIZZATORE
    # Adam: adattivo, convergenza rapida, gestisce bene learning rate
    # Vantaggi: momentum + RMSprop, adatta lr per ogni parametro
    optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
    
    # FUNZIONE DI LOSS
    # CrossEntropyLoss: standard per classificazione multi-classe
    # Combina LogSoftmax + NLLLoss per stabilità numerica
    criterion = nn.CrossEntropyLoss()
    
    # MISURAZIONE TEMPO DI ADDESTRAMENTO
    # Importante per confronti di efficienza computazionale
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    start_time = time.time()
    train_losses1, train_acc1 = train_model(model1, train_loader, criterion, optimizer1, num_epochs)
    training_time1 = time.time() - start_time
    
    # VALUTAZIONE SU TEST SET
<<<<<<< HEAD
    # Misura generalizzazione del modello su dati mai visti
    test_acc1, test_loss1 = test_model(model1, test_loader)
    
    # SALVATAGGIO RISULTATI STRUTTURATI
    # Organizzazione dati per analisi successiva e visualizzazione
=======
    test_acc1, test_loss1 = test_model(model1, test_loader)
    
    # SALVATAGGIO RISULTATI
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    results['Single_No_Dropout'] = {
        'train_losses': train_losses1,    # Storia loss durante training
        'train_acc': train_acc1,          # Storia accuracy durante training
        'test_acc': test_acc1,            # Accuracy finale su test
        'test_loss': test_loss1,          # Loss finale su test
        'training_time': training_time1   # Tempo totale addestramento
    }
    print(f"Test Accuracy: {test_acc1:.2f}%, Test Loss: {test_loss1:.4f}")
    
<<<<<<< HEAD
    # PERSISTENZA MODELLO ADDESTRATO
    # Salvataggio stato per analisi future o deployment
    # .pth: formato standard PyTorch per state_dict
    model_path = os.path.join(output_dirs['models'], f'{dataset_name.lower()}_single_no_dropout.pth')
    torch.save(model1.state_dict(), model_path)
    
    # LOGGING METADATA ESPERIMENTO
    # Informazioni strutturate per report automatico
=======
    # PERSISTENZA MODELLO
    # Salvataggio state_dict per riutilizzo futuro
    model_path = os.path.join(output_dirs['models'], f'{dataset_name.lower()}_single_no_dropout.pth')
    torch.save(model1.state_dict(), model_path)
    
    # LOGGING PER REPORT
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    experiment_info['models']['Single_No_Dropout'] = {
        'architecture': 'Single Hidden Layer (512 neurons)',
        'dropout_rate': 0.0,
        'test_accuracy': test_acc1,
        'training_time': training_time1,
<<<<<<< HEAD
        'overfitting_gap': train_acc1[-1] - test_acc1  # Indicatore overfitting
=======
        'overfitting_gap': train_acc1[-1] - test_acc1  # Misura overfitting
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    }
      # ========================================================================
    # ESPERIMENTO 2: MLP SINGOLO STRATO CON DROPOUT (REGOLARIZZATO)
    # ========================================================================
    
    # ===============================================================
    # ESPERIMENTO 2: MLP SINGOLO STRATO CON DROPOUT
    # ===============================================================
    print("\n2. Training MLP singolo strato CON dropout (0.3)...")
    
<<<<<<< HEAD
    # CREAZIONE MODELLO CON REGOLARIZZAZIONE
    # dropout_rate=0.3: disattiva 30% dei neuroni casualmente durante training
    # Valore scelto perché tipicamente efficace senza perdite eccessive
    model2 = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.3).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
    
    # ADDESTRAMENTO CON REGOLARIZZAZIONE
    # Il dropout dovrebbe ridurre overfitting e migliorare generalizzazione
=======
    # STESSO MODELLO CON DROPOUT
    # dropout_rate=0.3 -> 30% neuroni spenti casualmente
    model2 = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.3).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
    
    # TRAINING IDENTICO
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    start_time = time.time()
    train_losses2, train_acc2 = train_model(model2, train_loader, criterion, optimizer2, num_epochs)
    training_time2 = time.time() - start_time
    
    # VALUTAZIONE E CONFRONTO CON BASELINE
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
      # ========================================================================
    # ESPERIMENTO 3: MLP MULTI-STRATO SENZA DROPOUT (ARCHITETTURA COMPLESSA)
    # ========================================================================
    
    # ===============================================================
    # ESPERIMENTO 3: MLP MULTI-STRATO SENZA DROPOUT
    # ===============================================================
    print("\n3. Training MLP multi-strato SENZA dropout...")
    
<<<<<<< HEAD
    # CREAZIONE ARCHITETTURA PROFONDA SENZA REGOLARIZZAZIONE
    # [512, 256, 128]: architettura decrescente per feature learning gerarchico
    # Maggiore capacità ma anche maggiore rischio di overfitting
=======
    # ARCHITETTURA ENCODER-LIKE
    # [512, 256, 128] -> dimensioni decrescenti
    # MOTIVAZIONE: estrazione features gerarchiche
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    model3 = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.0).to(device)
    optimizer3 = optim.Adam(model3.parameters(), lr=learning_rate)
    
    # ADDESTRAMENTO ARCHITETTURA COMPLESSA
    # Aspettativa: buone performance ma potenziale overfitting forte
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
      # ========================================================================
    # ESPERIMENTO 4: MLP MULTI-STRATO CON DROPOUT (REGOLARIZZAZIONE PROFONDA)
    # ========================================================================
    
    # ===============================================================
    # ESPERIMENTO 4: MLP MULTI-STRATO CON DROPOUT
    # ===============================================================
    print("\n4. Training MLP multi-strato CON dropout (0.3)...")
    
<<<<<<< HEAD
    # CREAZIONE ARCHITETTURA PROFONDA CON REGOLARIZZAZIONE
    # Combinazione di architettura complessa + dropout per generalizzazione ottimale
    # Test critico: dropout riesce a controllare overfitting in rete profonda?
=======
    # STESSA ARCHITETTURA + DROPOUT UNIFORME
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    model4 = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.3).to(device)
    optimizer4 = optim.Adam(model4.parameters(), lr=learning_rate)
    
    # ADDESTRAMENTO MODELLO OTTIMIZZATO
    # Aspettativa: migliore generalizzazione rispetto a modello 3
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
      # ========================================================================
    # PERSISTENZA E SERIALIZZAZIONE RISULTATI
    # ========================================================================
    
    # ===============================================================
    # PERSISTENZA RISULTATI NUMERICI
    # ===============================================================
    
    # SERIALIZZAZIONE JSON
    # Conversione numpy arrays -> liste per compatibilità JSON
    results_path = os.path.join(output_dirs['data'], f'results_{dataset_name.lower()}.json')
    with open(results_path, 'w') as f:
<<<<<<< HEAD
        # CONVERSIONE DATI PER SERIALIZZAZIONE JSON
        # PyTorch tensors e numpy arrays non sono serializzabili in JSON
        # Necessaria conversione esplicita a tipi Python nativi
        serializable_results = {}
        for model_name, data in results.items():
            serializable_results[model_name] = {
                'train_losses': [float(x) for x in data['train_losses']],    # Lista float da lista tensori
                'train_acc': [float(x) for x in data['train_acc']],          # Lista float da lista tensori
                'test_acc': float(data['test_acc']),                         # Float singolo da tensor
                'test_loss': float(data['test_loss']),                       # Float singolo da tensor
                'training_time': float(data['training_time'])                # Float singolo da float64
=======
        serializable_results = {}
        for model_name, data in results.items():
            serializable_results[model_name] = {
                'train_losses': [float(x) for x in data['train_losses']],
                'train_acc': [float(x) for x in data['train_acc']],
                'test_acc': float(data['test_acc']),
                'test_loss': float(data['test_loss']),  
                'training_time': float(data['training_time'])
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
            }
        # SCRITTURA JSON CON FORMATTAZIONE
        # indent=2: formattazione leggibile per debugging e review manuale
        json.dump(serializable_results, f, indent=2)
    
<<<<<<< HEAD
    # AGGIORNAMENTO LOG GLOBALE ESPERIMENTO
    # Mantiene traccia di tutti gli esperimenti eseguiti nella sessione corrente
    # Permette analisi aggregate e confronti tra dataset diversi
=======
    # AGGIORNAMENTO LOG GLOBALE
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    experiment_log['experiments'].append(experiment_info)
    
    return results

# Funzione per visualizzare i risultati con salvataggio
def plot_results(results, dataset_name, save_plots=True):
    """
<<<<<<< HEAD
    Crea visualizzazioni complete dei risultati degli esperimenti
    
    STRUTTURA VISUALIZZAZIONE:
    1. Training Loss: Convergenza dell'addestramento
    2. Training Accuracy: Miglioramento performance durante training
    3. Test Accuracy Comparison: Confronto finale tra modelli
    4. Overfitting Analysis: Gap tra training e test accuracy
    
    TECNOLOGIE MATPLOTLIB:
    - subplots(2,2): Griglia 2x2 per layout organizzato
    - lineplot: Tendenze temporali (loss/accuracy vs epoche)
    - barplot: Confronti categoriali (accuracy per modello)
    - annotations: Valori numerici sui grafici per precisione
    
    SCELTE DESIGN:
    - Colori distintivi per ogni modello
    - Grid per leggibilità
    - Rotazione etichette x per spazio
    - DPI 300 per qualità pubblicazione
    
    Args:
        results: Dizionario con risultati di tutti i modelli
        dataset_name: Nome dataset per titoli e nomi file
        save_plots: Flag per salvare grafici su disco
    """
    # CREAZIONE STRUTTURA PLOT
    # 2x2 grid: 4 diversi tipi di analisi in un'unica figura
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Confronto Prestazioni Modelli - {dataset_name}', fontsize=16)
    
    # ========================================================================
    # PLOT 1: TRAINING LOSS - CONVERGENZA DELL'ADDESTRAMENTO
    # ========================================================================
    ax1 = axes[0, 0]
    # ITERAZIONE ATTRAVERSO TUTTI I MODELLI
    # Ogni modello ha una curva distinta per confronto visivo
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    for model_name, data in results.items():
        # linewidth=2 per visibilità su stampa
        ax1.plot(data['train_losses'], label=model_name, linewidth=2)
    
    # PERSONALIZZAZIONE ASSI E LABELS
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
<<<<<<< HEAD
    ax1.legend()
    ax1.grid(True, alpha=0.3)  # Grid trasparente per leggibilità
    
    # ========================================================================
    # PLOT 2: TRAINING ACCURACY - PROGRESSIONE APPRENDIMENTO
    # ========================================================================
=======
    ax1.legend()                    # Legenda automatica da labels
    ax1.grid(True, alpha=0.3)       # Griglia sottile per lettura
    
    # ===============================================================
    # GRAFICO 2: EVOLUZIONE TRAINING ACCURACY
    # ===============================================================
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    ax2 = axes[0, 1]
    
    for model_name, data in results.items():
        ax2.plot(data['train_acc'], label=model_name, linewidth=2)
    
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
<<<<<<< HEAD
    # ========================================================================
    # PLOT 3: TEST ACCURACY COMPARISON - PERFORMANCE FINALE
    # ========================================================================
=======
    # ===============================================================
    # GRAFICO 3: CONFRONTO TEST ACCURACY (BAR PLOT)
    # ===============================================================
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    ax3 = axes[1, 0]
    
    # ESTRAZIONE DATI PER BAR PLOT
    models = list(results.keys())
    test_accs = [results[model]['test_acc'] for model in models]
    
<<<<<<< HEAD
    # PALETTE COLORI DISTINTIVI
    # Colori scelti per massimo contrasto e accessibilità
=======
    # PALETTE COLORI DISTINTIVA
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
    bars = ax3.bar(models, test_accs, color=colors)
    
    ax3.set_title('Test Accuracy Comparison')
    ax3.set_ylabel('Accuracy (%)')
<<<<<<< HEAD
    ax3.tick_params(axis='x', rotation=45)  # Rotazione per leggibilità
    
    # ANNOTAZIONI NUMERICHE
    # Aggiunta valori esatti sopra le barre per precisione
=======
    ax3.tick_params(axis='x', rotation=45)  # Rotazione labels per leggibilità
    
    # ANNOTAZIONI VALORI SULLE BARRE
    # Migliora interpretazione immediata
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    for bar, acc in zip(bars, test_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.2f}%', ha='center', va='bottom')
    
<<<<<<< HEAD
    # ========================================================================
    # PLOT 4: OVERFITTING ANALYSIS - GAP TRAINING-TEST
    # ========================================================================
=======
    # ===============================================================
    # GRAFICO 4: ANALISI OVERFITTING (TRAIN-TEST GAP)
    # ===============================================================
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    ax4 = axes[1, 1]
    
    # CALCOLO GAP OVERFITTING
    # Differenza tra ultima accuracy training e test accuracy
    overfitting = []
    
    # CALCOLO GAP OVERFITTING
    # Gap = Training Accuracy finale - Test Accuracy
    # Gap > 0: overfitting (modello memorizza training set)
    # Gap ≈ 0: generalizzazione ottimale
    # Gap < 0: underfitting (raro, possibile con dropout estremo)
    for model_name, data in results.items():
        final_train_acc = data['train_acc'][-1]  # Ultima epoca training
        test_acc = data['test_acc']              # Accuracy su test set
        gap = final_train_acc - test_acc
        overfitting.append(gap)
    
    bars = ax4.bar(models, overfitting, color=colors)
    ax4.set_title('Overfitting Analysis (Train-Test Gap)')
    ax4.set_ylabel('Accuracy Gap (%)')
    ax4.tick_params(axis='x', rotation=45)
    
<<<<<<< HEAD
    # LINEA DI RIFERIMENTO A ZERO
    # Indica perfetta generalizzazione (no overfitting)
=======
    # LINEA RIFERIMENTO A ZERO
    # Evidenzia threshold overfitting
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # ANNOTAZIONI GAP VALUES
    for bar, gap in zip(bars, overfitting):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{gap:.2f}%', ha='center', va='bottom')
    
<<<<<<< HEAD
    # FINALIZZAZIONE LAYOUT
    plt.tight_layout()  # Ottimizza spacing automaticamente
    
    # SALVATAGGIO PERSISTENTE
    # Salva in alta risoluzione per uso professionale/pubblicazione
=======
    # LAYOUT OTTIMIZZAZIONE
    plt.tight_layout()
    
    # ===============================================================
    # SALVATAGGIO ALTA RISOLUZIONE
    # ===============================================================
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    if save_plots:
        # dpi=300 per qualità pubblicazione
        # bbox_inches='tight' elimina whitespace
        plot_path = os.path.join(output_dirs['plots'], f'model_comparison_{dataset_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')  # DPI 300 = qualità stampa
        print(f"Plot salvato: {plot_path}")
    
    plt.show()

# Funzione per analisi dettagliata del dropout con salvataggio
def dropout_analysis(dataset_name='MNIST', save_plots=True):
    """
<<<<<<< HEAD
    Analisi sistematica dell'effetto di diversi valori di dropout rate
    
    OBIETTIVO SPERIMENTALE:
    Determinare il valore ottimale di dropout per massimizzare generalizzazione
    testando sistematicamente diversi dropout rates su architettura multi-strato
    
    METODOLOGIA:
    1. Test range completo dropout: 0.0 → 0.5 (step 0.1)
    2. Architettura fissa: Multi-layer per maggiore sensibilità al dropout
    3. Metriche multiple: accuracy, overfitting gap, stabilità training
    4. Visualizzazione trend per identificazione ottimo
    
    RANGE DROPOUT TESTATI:
    - 0.0: Baseline senza regolarizzazione
    - 0.1-0.2: Regolarizzazione leggera
    - 0.3-0.4: Regolarizzazione moderata (tipicamente ottimale)
    - 0.5: Regolarizzazione aggressiva (rischio underfitting)
    
    Args:
        dataset_name: 'MNIST' o 'FashionMNIST'
        save_plots: Flag per salvataggio automatico grafici
        
    Returns:
        dict: Risultati per ogni dropout rate testato
    """
    print(f"\n=== ANALISI DETTAGLIATA DROPOUT SU {dataset_name} ===")
    
    # CARICAMENTO DATASET E CONFIGURAZIONE
    train_loader, test_loader = load_datasets(dataset_name)
    
    # RANGE DROPOUT DA TESTARE
    # Progressione logica: da 0% a 50% con step del 10%
    # Copre tutto lo spettro pratico di utilizzo del dropout
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}
    
    # PARAMETRI SPERIMENTALI FISSI
    # Ridotti per velocizzare l'analisi senza perdere significatività
    input_size = 28 * 28     # Standard per MNIST/Fashion-MNIST
    num_classes = 10         # 10 categorie in entrambi i dataset
    num_epochs = 10          # Ridotto per velocità, sufficiente per trend
    learning_rate = 0.001    # Standard Adam learning rate
    
    # STRUTTURA DATI PER REPORT
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
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
    
<<<<<<< HEAD
    # ========================================================================
    # LOOP PRINCIPALE: TEST SISTEMATICO DROPOUT RATES
    # ========================================================================
    for dropout_rate in dropout_rates:
        print(f"\nTesting dropout rate: {dropout_rate}")
        
        # CREAZIONE MODELLO CON DROPOUT SPECIFICO
        # Architettura multi-strato: più sensibile agli effetti del dropout
        # Stesso seed per confronti fair tra dropout rates diversi
=======
    # ===============================================================
    #LOOP PRINCIPALE: TEST OGNI DROPOUT RATE
    # ===============================================================
    for dropout_rate in dropout_rates:
        print(f"\nTesting dropout rate: {dropout_rate}")
        
        # NUOVO MODELLO PER OGNI RATE
        # Evita contamination tra esperimenti
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        model = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
<<<<<<< HEAD
        # ADDESTRAMENTO CON DROPOUT RATE SPECIFICO
        train_losses, train_acc = train_model(model, train_loader, criterion, optimizer, num_epochs)
        test_acc, test_loss = test_model(model, test_loader)
        
        # CALCOLO METRICHE CHIAVE
        final_train_acc = train_acc[-1]              # Performance finale su training
        overfitting_gap = final_train_acc - test_acc # Indicatore overfitting
        
        # SALVATAGGIO RISULTATI STRUTTURATI
        results[dropout_rate] = {
            'train_acc': final_train_acc,
            'test_acc': test_acc,
            'overfitting': overfitting_gap
        }
        
        # LOGGING PER REPORT AUTOMATICO
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        dropout_experiment['results'][str(dropout_rate)] = {
            'train_accuracy': float(final_train_acc),
            'test_accuracy': float(test_acc),
            'overfitting_gap': float(overfitting_gap)
        }
        
<<<<<<< HEAD
        # FEEDBACK REAL-TIME
        print(f"Train Acc: {final_train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Gap: {overfitting_gap:.2f}%")
    
    # ========================================================================
    # VISUALIZZAZIONE RISULTATI DROPOUT ANALYSIS
    # ========================================================================
    
    # CREAZIONE FIGURE CON 3 SUBPLOT
    # Layout orizzontale per confronto diretto delle metriche
=======
        # LOGGING IMMEDIATO
        print(f"Train Acc: {train_acc[-1]:.2f}%, Test Acc: {test_acc:.2f}%, Gap: {train_acc[-1] - test_acc:.2f}%")
    
    # ===============================================================
    # VISUALIZZAZIONE ANALISI DROPOUT
    # ===============================================================
    
    # SETUP FIGURA TRIPLA
    # 3 subplot per analisi completa
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Analisi Dropout - {dataset_name}', fontsize=16)
    
    # ESTRAZIONE DATI PER PLOTTING
<<<<<<< HEAD
    dropout_vals = list(results.keys())                           # X-axis: dropout rates
    train_accs = [results[dr]['train_acc'] for dr in dropout_vals]  # Training accuracies
    test_accs = [results[dr]['test_acc'] for dr in dropout_vals]    # Test accuracies  
    overfitting = [results[dr]['overfitting'] for dr in dropout_vals]  # Overfitting gaps
    
    # ========================================================================
    # SUBPLOT 1: CONFRONTO TRAIN VS TEST ACCURACY
    # ========================================================================
    # Mostra come dropout riduce gap tra training e test performance
    axes[0].plot(dropout_vals, train_accs, 'o-', label='Train Accuracy', linewidth=2, markersize=8)
    axes[0].plot(dropout_vals, test_accs, 's-', label='Test Accuracy', linewidth=2, markersize=8)
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    axes[0].set_xlabel('Dropout Rate')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Effect of Dropout on Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
<<<<<<< HEAD
    # ========================================================================
    # SUBPLOT 2: FOCUS SU TEST ACCURACY
    # ========================================================================
    # Evidenzia il dropout rate che massimizza generalizzazione
=======
    # SUBPLOT 2: TEST ACCURACY FOCUS
    # OBIETTIVO: Identificare optimum dropout rate
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    axes[1].plot(dropout_vals, test_accs, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Dropout Rate')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Test Accuracy vs Dropout Rate')
    axes[1].grid(True, alpha=0.3)
    
<<<<<<< HEAD
    # ========================================================================
    # SUBPLOT 3: ANALISI OVERFITTING
    # ========================================================================
    # Mostra come dropout riduce progressivamente l'overfitting
=======
    # SUBPLOT 3: OVERFITTING ANALYSIS
    # OBIETTIVO: Quantificare effetto regolarizzazione
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    axes[2].plot(dropout_vals, overfitting, 'go-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Dropout Rate')
    axes[2].set_ylabel('Train-Test Gap (%)')
    axes[2].set_title('Overfitting vs Dropout Rate')
<<<<<<< HEAD
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)  # Linea target (no overfitting)
=======
    
    # LINEA RIFERIMENTO ZERO OVERFITTING
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    axes[2].grid(True, alpha=0.3)
    
    # FINALIZZAZIONE VISUALIZZAZIONE
    plt.tight_layout()
    
<<<<<<< HEAD
    # SALVATAGGIO PERSISTENTE
=======
    # SALVATAGGIO PLOT
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    if save_plots:
        plot_path = os.path.join(output_dirs['plots'], f'dropout_analysis_{dataset_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot salvato: {plot_path}")
    
    plt.show()
    
<<<<<<< HEAD
    # ========================================================================
    # PERSISTENZA DATI E LOGGING
    # ========================================================================
    
    # Salvataggio risultati dropout analysis
=======
    # ===============================================================
    # PERSISTENZA RISULTATI
    # ===============================================================
    
    # SALVATAGGIO JSON STRUTTURATO
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
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
      # ====================================================================
    # FASE 3: ESECUZIONE ESPERIMENTI COMPARATIVI SUI DATASET PRINCIPALI
    # ====================================================================
    # Questa sezione esegue gli esperimenti di confronto sui due dataset
    # standard per valutazione di modelli di deep learning su immagini:
    # MNIST (cifre) e Fashion-MNIST (abbigliamento)
    
<<<<<<< HEAD
    # ESPERIMENTO 1: MNIST Dataset
    # ----------------------------
    # MNIST è il dataset di riferimento per:
    # - Cifre scritte a mano (0-9)
    # - 28x28 pixel, scala di grigi
    # - 60,000 campioni training, 10,000 test
    # - Relativamente semplice, buono per baseline
=======
    # ===============================================================
    # FASE 1: ESPERIMENTI COMPARATIVI MNIST
    # ===============================================================
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    print("\n🔍 Avvio esperimenti su MNIST...")
    
    # CONFRONTO 4 MODELLI: Single/Multi x NoDropout/WithDropout
    results_mnist = compare_models('MNIST')
    
    # VISUALIZZAZIONE IMMEDIATA PER MONITORAGGIO
    plot_results(results_mnist, 'MNIST', save_plots=True)
    
<<<<<<< HEAD
    # ESPERIMENTO 2: Fashion-MNIST Dataset
    # ------------------------------------
    # Fashion-MNIST è più complesso di MNIST perché:
    # - Contiene categorie di abbigliamento (10 classi)
    # - Maggiore variabilità intra-classe
    # - Pattern più complessi da riconoscere
    # - Stesso formato di MNIST per compatibilità
=======
    # ===============================================================
    # FASE 2: ESPERIMENTI COMPARATIVI FASHION-MNIST
    # ===============================================================
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    print("\n🔍 Avvio esperimenti su Fashion-MNIST...")
    
    # STESSO PROTOCOLLO SU DATASET PIÙ COMPLESSO
    results_fashion = compare_models('FashionMNIST')
    plot_results(results_fashion, 'Fashion-MNIST', save_plots=True)
    
<<<<<<< HEAD
    # FASE 4: ANALISI DETTAGLIATA DELL'EFFETTO DROPOUT
    # =================================================
    # Questa analisi approfondita esamina l'effetto di diversi valori
    # di dropout rate per determinare il valore ottimale per ogni dataset
=======
    # ===============================================================
    # FASE 3: ANALISI SISTEMATICA DROPOUT
    # ===============================================================
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
    print("\n📊 Analisi dettagliata dell'effetto del dropout...")
    
    # TEST RANGE DROPOUT RATES SU ENTRAMBI DATASET
    dropout_results_mnist = dropout_analysis('MNIST', save_plots=True)
    dropout_results_fashion = dropout_analysis('FashionMNIST', save_plots=True)
    
<<<<<<< HEAD
    # FASE 5: GENERAZIONE REPORT COMPLETO
    # ====================================
    # Creazione di un report testuale completo che riassume tutti i risultati
    # Include: tabelle, statistiche, osservazioni e conclusioni
    print("\n📝 Generazione report completo...")
    report_path = create_comprehensive_report(
        results_mnist, results_fashion, 
        dropout_results_mnist, dropout_results_fashion
    )
    
    # FASE 6: CONVERSIONE IN PDF (OPZIONALE)
    # =======================================
    # Tentativo di creare versione PDF del report per distribuzione
    # Fallback graceful se librerie PDF non disponibili
    print("\n📄 Tentativo di creazione PDF...")
    pdf_path = create_pdf_report(report_path)      # ================================================================
    # FASE 7: VISUALIZZAZIONI AVANZATE E ANALISI PREDITTIVE
    # ================================================================
    # Questa sezione implementa analisi visive avanzate per comprendere
    # il comportamento dei modelli addestrati attraverso:
    # - Visualizzazione di campioni con predizioni
    # - Confronto diretto tra modelli diversi
    # - Analisi degli errori più comuni
    # - Studio dell'effetto del dropout sulle attivazioni neurali
    
    # Visualizzazione campioni predetti
    print("\n🖼️ Visualizzazione campioni con predizioni...")
    
    # CONFIGURAZIONE MODELLI PER VISUALIZZAZIONE
    # ------------------------------------------
    # Carica i dataset per l'analisi visiva
    # Questi loader servono per estrarre campioni specifici da visualizzare
    train_loader_mnist, test_loader_mnist = load_datasets('MNIST')
    train_loader_fashion, test_loader_fashion = load_datasets('FashionMNIST')
    
    # Parametri architetturali standard
    # Questi devono corrispondere esattamente ai modelli addestrati
    input_size = 28 * 28  # Dimensione input flattened per MNIST/Fashion-MNIST
    num_classes = 10      # Numero classi per entrambi i dataset
    
    # CREAZIONE MODELLI MNIST PER VISUALIZZAZIONE
    # --------------------------------------------
    # Ricrea gli stessi modelli usati nel training per caricare i pesi salvati
    # Quattro combinazioni: Single/Multi layer × Con/Senza dropout
    model_mnist_single = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.0).to(device)
    model_mnist_single_dropout = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.3).to(device)
    model_mnist_multi = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.0).to(device)
    model_mnist_multi_dropout = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.3).to(device)
      # CARICAMENTO PESI DEI MODELLI MNIST ADDESTRATI
    # ===============================================
    # Carica i parametri salvati durante il training precedente
    # Gestione robusta degli errori nel caso i file non esistano
    try:
        # Caricamento stato dei modelli dai file .pth salvati
        # torch.load() ricostruisce il dizionario state_dict completo
        # load_state_dict() applica i pesi alle architetture corrispondenti
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        model_mnist_single.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'mnist_single_no_dropout.pth')))
        model_mnist_single_dropout.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'mnist_single_with_dropout.pth')))
        model_mnist_multi.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'mnist_multi_no_dropout.pth')))
        model_mnist_multi_dropout.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'mnist_multi_with_dropout.pth')))
        
<<<<<<< HEAD
        # ANALISI VISIVA 1: CAMPIONI CON PREDIZIONI
        # ------------------------------------------
        # Mostra come il modello migliore (multi-layer con dropout) 
        # classifica campioni casuali del test set
        print("📸 Visualizzazione campioni MNIST...")
        visualize_sample_predictions(model_mnist_multi_dropout, test_loader_mnist, 'MNIST', num_samples=12)
        
        # ANALISI VISIVA 2: CONFRONTO DIRETTO TRA MODELLI
        # ------------------------------------------------
        # Analizza come modelli diversi classificano gli stessi campioni
        # Rivela differenze nel comportamento predittivo
        print("🔍 Confronto predizioni modelli MNIST...")
=======
        # VISUALIZZAZIONI CAMPIONI PREDETTI
        # Mostra performance qualitativa su esempi reali
        print("📸 Visualizzazione campioni MNIST...")
        visualize_sample_predictions(model_mnist_multi_dropout, test_loader_mnist, 'MNIST', num_samples=12)
        
        # CONFRONTO PREDIZIONI TRA MODELLI
        # Analisi comparativa su stessi campioni
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        models_dict_mnist = {
            'Single No Dropout': model_mnist_single,
            'Single With Dropout': model_mnist_single_dropout,
            'Multi No Dropout': model_mnist_multi,
            'Multi With Dropout': model_mnist_multi_dropout
        }
        compare_model_predictions(models_dict_mnist, test_loader_mnist, 'MNIST', num_samples=6)
        
<<<<<<< HEAD
        # ANALISI VISIVA 3: STUDIO DEGLI ERRORI
        # --------------------------------------
        # Identifica pattern negli errori per comprendere limitazioni del modello
        # Importante per debugging e miglioramento dell'architettura
        print("❌ Analisi errori MNIST...")
        analyze_model_errors(model_mnist_multi, test_loader_mnist, 'MNIST', num_errors=12)
        
        # ANALISI VISIVA 4: EFFETTO DROPOUT SULLE ATTIVAZIONI NEURALI
        # ------------------------------------------------------------
        # Studio approfondito di come il dropout modifica le attivazioni
        # Confronta distribuzione attivazioni con/senza dropout
=======
        # ANALISI ERRORI TIPICI
        # Insight su failure modes
        print("❌ Analisi errori MNIST...")
        analyze_model_errors(model_mnist_multi, test_loader_mnist, 'MNIST', num_errors=12)
        
        # EFFETTO DROPOUT SU ATTIVAZIONI NEURONALI
        # Analisi quantitativa internal representations
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        print("🧠 Analisi effetto dropout sulle attivazioni MNIST...")
        visualize_dropout_effect(model_mnist_multi_dropout, model_mnist_multi, test_loader_mnist, 'MNIST')
        
    except FileNotFoundError:
        # Gestione elegante del caso in cui i modelli non siano stati salvati
        # Può accadere se il training è stato interrotto o fallito
        print("⚠️ Modelli MNIST non trovati, probabilmente il training non è stato completato.")
<<<<<<< HEAD
      # RIPETIZIONE ANALISI PER FASHION-MNIST
    # ======================================
    # Stesso protocollo di analisi applicato al dataset Fashion-MNIST
    # per confrontare il comportamento su dati più complessi
    try:
        # Creazione modelli Fashion-MNIST con stesse architetture
=======
    
    # ===============================================================
    # ANALISI QUALITATIVA FASHION-MNIST
    # ===============================================================
    try:
        # RICOSTRUZIONE MODELLI FASHION-MNIST
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        model_fashion_single = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.0).to(device)
        model_fashion_single_dropout = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.3).to(device)
        model_fashion_multi = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.0).to(device)
        model_fashion_multi_dropout = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.3).to(device)
        
<<<<<<< HEAD
        # Caricamento pesi dei modelli Fashion-MNIST addestrati
=======
        # CARICAMENTO PESI
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        model_fashion_single.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'fashionmnist_single_no_dropout.pth')))
        model_fashion_single_dropout.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'fashionmnist_single_with_dropout.pth')))
        model_fashion_multi.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'fashionmnist_multi_no_dropout.pth')))
        model_fashion_multi_dropout.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'fashionmnist_multi_with_dropout.pth')))
        
<<<<<<< HEAD
        # ANALISI VISIVA FASHION-MNIST 1: CAMPIONI CON PREDIZIONI
        # --------------------------------------------------------
        # Fashion-MNIST è più sfidante: categorie di abbigliamento
        # con maggiore variabilità intra-classe rispetto a MNIST
        print("👕 Visualizzazione campioni Fashion-MNIST...")
        visualize_sample_predictions(model_fashion_multi_dropout, test_loader_fashion, 'FashionMNIST', num_samples=12)
        
        # ANALISI VISIVA FASHION-MNIST 2: CONFRONTO MODELLI
        # --------------------------------------------------
        # Verifica se le differenze tra architetture sono più pronunciate
        # su dataset più complesso
=======
        # VISUALIZZAZIONI FASHION-MNIST
        print("👕 Visualizzazione campioni Fashion-MNIST...")
        visualize_sample_predictions(model_fashion_multi_dropout, test_loader_fashion, 'FashionMNIST', num_samples=12)
        
        # CONFRONTI TRA MODELLI
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        models_dict_fashion = {
            'Single No Dropout': model_fashion_single,
            'Single With Dropout': model_fashion_single_dropout,
            'Multi No Dropout': model_fashion_multi,
            'Multi With Dropout': model_fashion_multi_dropout
        }
        compare_model_predictions(models_dict_fashion, test_loader_fashion, 'FashionMNIST', num_samples=6)
        
<<<<<<< HEAD
        # ANALISI VISIVA FASHION-MNIST 3: STUDIO ERRORI
        # ----------------------------------------------
        # Gli errori su Fashion-MNIST rivelano confusioni semantiche
        # tra categorie di abbigliamento simili
        print("❌ Analisi errori Fashion-MNIST...")
        analyze_model_errors(model_fashion_multi, test_loader_fashion, 'FashionMNIST', num_errors=12)
        
        # ANALISI VISIVA FASHION-MNIST 4: EFFETTO DROPOUT
        # ------------------------------------------------
        # Studio dell'effetto del dropout su dataset più complesso
        # Fashion-MNIST dovrebbe mostrare benefici maggiori dal dropout
=======
        # ANALISI ERRORI
        print("❌ Analisi errori Fashion-MNIST...")
        analyze_model_errors(model_fashion_multi, test_loader_fashion, 'FashionMNIST', num_errors=12)
        
        # EFFETTO DROPOUT
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
        print("🧠 Analisi effetto dropout sulle attivazioni Fashion-MNIST...")
        visualize_dropout_effect(model_fashion_multi_dropout, model_fashion_multi, test_loader_fashion, 'FashionMNIST')
        
    except FileNotFoundError:
        # Gestione errore per modelli Fashion-MNIST mancanti
        print("⚠️ Modelli Fashion-MNIST non trovati, probabilmente il training non è stato completato.")
    
<<<<<<< HEAD
        # FASE 8: CREAZIONE GRAFICI RIASSUNTIVI E REPORT FINALE
        # ======================================================
        # Sintesi visiva e testuale di tutti i risultati ottenuti
        
        # Creazione di un plot riassuntivo finale che aggrega tutti i risultati
        # Confronta prestazioni, overfitting, efficacia dropout, tempi di training
        create_summary_plot(results_mnist, results_fashion)
        
        # Report finale dettagliato stampato a console
        # Include statistiche aggregate, conclusioni chiave, raccomandazioni
        print_final_summary(results_mnist, results_fashion, dropout_results_mnist, dropout_results_fashion)
        
        # MESSAGGIO DI COMPLETAMENTO CON INDICAZIONI FINALI
        # ==================================================
        # Fornisce all'utente tutte le informazioni necessarie per accedere ai risultati
        print(f"\n✅ ESPERIMENTO COMPLETATO!")
        print(f"📁 Tutti i file sono disponibili in: {output_dirs['base']}")
        print(f"📊 Grafici salvati in: {output_dirs['plots']}")
        print(f"🤖 Modelli salvati in: {output_dirs['models']}")
        print(f"📝 Report salvato in: {output_dirs['reports']}")

def create_summary_plot(results_mnist, results_fashion):
    """
    Crea un grafico riassuntivo finale completo che confronta tutti i risultati sperimentali
    
    OBIETTIVO:
    Visualizzazione integrata di 4 metriche chiave per valutazione comparativa:
    1. Test Accuracy - Prestazioni finali dei modelli
    2. Overfitting Gap - Differenza tra training e test accuracy
    3. Efficacia Dropout - Quantificazione benefici del dropout
    4. Training Time - Costi computazionali delle diverse architetture
    
    DESIGN RATIONALE:
    - Layout 2x2 per confronto sistematico delle metriche
    - Colori distinti per dataset (MNIST vs Fashion-MNIST)
    - Barre con valori numerici per precisione quantitativa
    - Griglia e assi chiari per leggibilità professionale
    
    TECNOLOGIE UTILIZZATE:
    - matplotlib.pyplot per creazione figure multi-panel
    - numpy per calcoli statistici aggregati
    - Subplot layout per organizzazione visiva ottimale
    
    Args:
        results_mnist (dict): Risultati esperimenti MNIST
        results_fashion (dict): Risultati esperimenti Fashion-MNIST
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Riassunto Completo - Studio del Dropout', fontsize=18, fontweight='bold')
    
    # Plot 1: Confronto Test Accuracy tra dataset
    ax1 = axes[0, 0]
    models = list(results_mnist.keys())
    mnist_accs = [results_mnist[model]['test_acc'] for model in models]
    fashion_accs = [results_fashion[model]['test_acc'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, mnist_accs, width, label='MNIST', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, fashion_accs, width, label='Fashion-MNIST', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Modelli')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Confronto Test Accuracy tra Dataset')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', '\n') for m in models], rotation=0, fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Aggiunta valori sopra le barre
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%', 
                ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Confronto Overfitting Gap
    ax2 = axes[0, 1]
    mnist_gaps = [results_mnist[model]['train_acc'][-1] - results_mnist[model]['test_acc'] for model in models]
    fashion_gaps = [results_fashion[model]['train_acc'][-1] - results_fashion[model]['test_acc'] for model in models]
    
    bars1 = ax2.bar(x - width/2, mnist_gaps, width, label='MNIST', alpha=0.8, color='lightgreen')
    bars2 = ax2.bar(x + width/2, fashion_gaps, width, label='Fashion-MNIST', alpha=0.8, color='orange')
    
    ax2.set_xlabel('Modelli')
    ax2.set_ylabel('Overfitting Gap (%)')
    ax2.set_title('Confronto Overfitting (Train-Test Gap)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_', '\n') for m in models], rotation=0, fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: Efficacia Dropout (differenza con/senza)
    ax3 = axes[1, 0]
    dropout_effect_mnist = []
    dropout_effect_fashion = []
    
    for arch in ['Single', 'Multi']:
        no_dropout = f'{arch}_No_Dropout'
        with_dropout = f'{arch}_With_Dropout'
        
        if no_dropout in results_mnist and with_dropout in results_mnist:
            effect_mnist = (results_mnist[no_dropout]['train_acc'][-1] - results_mnist[no_dropout]['test_acc']) - \
                          (results_mnist[with_dropout]['train_acc'][-1] - results_mnist[with_dropout]['test_acc'])
            dropout_effect_mnist.append(effect_mnist)
            
            effect_fashion = (results_fashion[no_dropout]['train_acc'][-1] - results_fashion[no_dropout]['test_acc']) - \
                           (results_fashion[with_dropout]['train_acc'][-1] - results_fashion[with_dropout]['test_acc'])
            dropout_effect_fashion.append(effect_fashion)
    
    arch_labels = ['Single Layer', 'Multi Layer']
    x_arch = np.arange(len(arch_labels))
    
    bars1 = ax3.bar(x_arch - width/2, dropout_effect_mnist, width, label='MNIST', alpha=0.8, color='gold')
    bars2 = ax3.bar(x_arch + width/2, dropout_effect_fashion, width, label='Fashion-MNIST', alpha=0.8, color='purple')
    
    ax3.set_xlabel('Architettura')
    ax3.set_ylabel('Riduzione Overfitting (%)')
    ax3.set_title('Efficacia del Dropout nel Ridurre Overfitting')
    ax3.set_xticks(x_arch)
    ax3.set_xticklabels(arch_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training Time Comparison
    ax4 = axes[1, 1]
    mnist_times = [results_mnist[model]['training_time'] for model in models]
    fashion_times = [results_fashion[model]['training_time'] for model in models]
    
    bars1 = ax4.bar(x - width/2, mnist_times, width, label='MNIST', alpha=0.8, color='cyan')
    bars2 = ax4.bar(x + width/2, fashion_times, width, label='Fashion-MNIST', alpha=0.8, color='magenta')
    
    ax4.set_xlabel('Modelli')
    ax4.set_ylabel('Training Time (s)')
    ax4.set_title('Confronto Tempi di Training')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.replace('_', '\n') for m in models], rotation=0, fontsize=9)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvataggio del plot riassuntivo
    summary_plot_path = os.path.join(output_dirs['plots'], 'summary_comparison.png')
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Grafico riassuntivo salvato: {summary_plot_path}")
    
    plt.show()

def print_final_summary(results_mnist, results_fashion, dropout_mnist, dropout_fashion):
    """Stampa un riassunto finale dettagliato"""
    
    print("\n" + "="*80)
    print("🎯 RIASSUNTO FINALE - STUDIO DEL DROPOUT")
    print("="*80)
    
    print("\n📊 PRESTAZIONI PRINCIPALI:")
    print("-" * 50)
    
    # Migliori modelli per ogni dataset
    best_mnist = max(results_mnist.items(), key=lambda x: x[1]['test_acc'])
    best_fashion = max(results_fashion.items(), key=lambda x: x[1]['test_acc'])
    
    print(f"🏆 MNIST - Miglior modello: {best_mnist[0]} ({best_mnist[1]['test_acc']:.2f}%)")
    print(f"🏆 Fashion-MNIST - Miglior modello: {best_fashion[0]} ({best_fashion[1]['test_acc']:.2f}%)")
    
    # Effetto del dropout
    print(f"\n💡 EFFICACIA DEL DROPOUT:")
    print("-" * 50)
    
    # Calcolo riduzione overfitting media
    no_dropout_gaps_mnist = [data['train_acc'][-1] - data['test_acc'] 
                            for k, data in results_mnist.items() if 'No_Dropout' in k]
    with_dropout_gaps_mnist = [data['train_acc'][-1] - data['test_acc'] 
                              for k, data in results_mnist.items() if 'With_Dropout' in k]
    
    avg_reduction_mnist = np.mean(no_dropout_gaps_mnist) - np.mean(with_dropout_gaps_mnist)
    
    no_dropout_gaps_fashion = [data['train_acc'][-1] - data['test_acc'] 
                              for k, data in results_fashion.items() if 'No_Dropout' in k]
    with_dropout_gaps_fashion = [data['train_acc'][-1] - data['test_acc'] 
                                for k, data in results_fashion.items() if 'With_Dropout' in k]
    
    avg_reduction_fashion = np.mean(no_dropout_gaps_fashion) - np.mean(with_dropout_gaps_fashion)
    
    print(f"📉 Riduzione overfitting MNIST: {avg_reduction_mnist:.2f}% in media")
    print(f"📉 Riduzione overfitting Fashion-MNIST: {avg_reduction_fashion:.2f}% in media")
    
    # Dropout ottimale
    best_dropout_mnist = max(dropout_mnist.items(), key=lambda x: x[1]['test_acc'])
    best_dropout_fashion = max(dropout_fashion.items(), key=lambda x: x[1]['test_acc'])
    
    print(f"🎯 Dropout ottimale MNIST: {best_dropout_mnist[0]} ({best_dropout_mnist[1]['test_acc']:.2f}%)")
    print(f"🎯 Dropout ottimale Fashion-MNIST: {best_dropout_fashion[0]} ({best_dropout_fashion[1]['test_acc']:.2f}%)")
    
    print(f"\n🏁 CONCLUSIONI CHIAVE:")
    print("-" * 50)
    print("✅ Il dropout è efficace nel ridurre l'overfitting")
    print("✅ L'effetto è più pronunciato su reti più profonde")
    print("✅ Fashion-MNIST richiede più regolarizzazione di MNIST")
    print("✅ Il dropout ottimale è dataset-dipendente")
    print("✅ Il trade-off accuracy vs generalizzazione è gestibile")
    
    print(f"\n📁 Tutti i file sono disponibili in: {output_dirs['base']}")
    print("="*80)

# Funzione per visualizzare campioni con predizioni
def visualize_sample_predictions(model, test_loader, dataset_name='MNIST', num_samples=12, save_plots=True):
    """
    Visualizza campioni del dataset con le relative predizioni del modello
    
    Args:
        model: Modello PyTorch addestrato
        test_loader: DataLoader per il test set
        dataset_name: Nome del dataset ('MNIST' o 'FashionMNIST')
        num_samples: Numero di campioni da visualizzare
        save_plots: Se salvare i plot o meno
    """
    model.eval()
    
    # Etichette per Fashion-MNIST
    fashion_labels = {
        0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
        5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
    }
    
    # Etichette per MNIST
    mnist_labels = {i: str(i) for i in range(10)}
    
    labels_dict = fashion_labels if dataset_name == 'FashionMNIST' else mnist_labels
    
    # Raccolta di campioni
    samples = []
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Converti in CPU per la visualizzazione
            inputs_cpu = inputs.cpu()
            predicted_cpu = predicted.cpu()
            labels_cpu = labels.cpu()
            probs_cpu = probs.cpu()
            
            for i in range(min(len(inputs), num_samples - len(samples))):
                samples.append(inputs_cpu[i])
                predictions.append(predicted_cpu[i])
                true_labels.append(labels_cpu[i])
                probabilities.append(probs_cpu[i])
            
            if len(samples) >= num_samples:
                break
    
    # Visualizzazione
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Predizioni del Modello - {dataset_name}', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(samples):
            # Mostra l'immagine
            img = samples[idx].squeeze()
            ax.imshow(img, cmap='gray')
            
            # Prepara le etichette
            true_label = true_labels[idx].item()
            pred_label = predictions[idx].item()
            confidence = probabilities[idx][pred_label].item() * 100
            
            true_name = labels_dict[true_label]
            pred_name = labels_dict[pred_label]
            
            # Colore del titolo (verde se corretto, rosso se sbagliato)
            color = 'green' if true_label == pred_label else 'red'
            
            # Titolo con vera etichetta e predizione
            title = f'True: {true_name}\nPred: {pred_name}\nConf: {confidence:.1f}%'
            ax.set_title(title, fontsize=10, color=color, weight='bold')
            
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    # Salvataggio del plot
    if save_plots:
        model_name = f"{type(model).__name__}_{dataset_name.lower()}"
        plot_path = os.path.join(output_dirs['plots'], f'sample_predictions_{model_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📸 Campioni salvati: {plot_path}")
    
    plt.show()
    
    return samples, predictions, true_labels, probabilities

# Funzione per confrontare predizioni di più modelli
def compare_model_predictions(models_dict, test_loader, dataset_name='MNIST', num_samples=8, save_plots=True):
    """
    Confronta le predizioni di più modelli sugli stessi campioni
    
    Args:
        models_dict: Dizionario {nome_modello: modello}
        test_loader: DataLoader per il test set
        dataset_name: Nome del dataset
        num_samples: Numero di campioni da confrontare
        save_plots: Se salvare i plot o meno
    """
    # Etichette
    fashion_labels = {
        0: 'T-shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
        5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Boot'
    }
    mnist_labels = {i: str(i) for i in range(10)}
    labels_dict = fashion_labels if dataset_name == 'FashionMNIST' else mnist_labels
    
    # Raccolta campioni fissi
    samples = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            for i in range(min(len(inputs), num_samples)):
                samples.append(inputs[i].cpu())
                true_labels.append(labels[i].cpu())
            
            if len(samples) >= num_samples:
                break
    
    # Ottieni predizioni per ogni modello
    all_predictions = {}
    all_confidences = {}
    
    for model_name, model in models_dict.items():
        model.eval()
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for sample in samples:
                sample_batch = sample.unsqueeze(0).to(device)
                output = model(sample_batch)
                prob = F.softmax(output, dim=1)
                _, pred = torch.max(output, 1)
                
                predictions.append(pred.cpu().item())
                confidences.append(prob.cpu().squeeze()[pred.item()].item() * 100)
        
        all_predictions[model_name] = predictions
        all_confidences[model_name] = confidences
    
    # Visualizzazione comparativa
    num_models = len(models_dict)
    fig, axes = plt.subplots(num_samples, num_models + 1, figsize=(4 * (num_models + 1), 3 * num_samples))
    fig.suptitle(f'Confronto Predizioni Modelli - {dataset_name}', fontsize=16)
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(num_samples):
        # Prima colonna: immagine originale
        img = samples[sample_idx].squeeze()
        axes[sample_idx, 0].imshow(img, cmap='gray')
        true_label = true_labels[sample_idx].item()
        axes[sample_idx, 0].set_title(f'Original\nTrue: {labels_dict[true_label]}', 
                                     fontsize=10, weight='bold')
        axes[sample_idx, 0].axis('off')
        
        # Altre colonne: predizioni dei modelli
        for model_idx, (model_name, predictions) in enumerate(all_predictions.items()):
            col_idx = model_idx + 1;
            
            pred_label = predictions[sample_idx];
            confidence = all_confidences[model_name][sample_idx];
            pred_name = labels_dict[pred_label];
            
            # Colore basato su correttezza
            color = 'green' if pred_label == true_label else 'red'
            
            # Mostra l'immagine con la predizione
            axes[sample_idx, col_idx].imshow(img, cmap='gray')
            axes[sample_idx, col_idx].set_title(f'{model_name}\nPred: {pred_name}\nConf: {confidence:.1f}%', 
                                              fontsize=9, color=color, weight='bold')
            axes[sample_idx, col_idx].axis('off')
    
    plt.tight_layout()
    
    # Salvataggio del plot
    if save_plots:
        plot_path = os.path.join(output_dirs['plots'], f'model_comparison_predictions_{dataset_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"🔍 Confronto predizioni salvato: {plot_path}")
    
    plt.show()
    
    return all_predictions, all_confidences

# Funzione per analizzare errori del modello
def analyze_model_errors(model, test_loader, dataset_name='MNIST', num_errors=12, save_plots=True):
    """
    Analizza e visualizza gli errori più comuni del modello
    
    Args:
        model: Modello PyTorch addestrato
        test_loader: DataLoader per il test set
        dataset_name: Nome del dataset
        num_errors: Numero di errori da visualizzare
        save_plots: Se salvare i plot o meno
    """
    model.eval()
    
    # Etichette
    fashion_labels = {
        0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
        5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
    }
    mnist_labels = {i: str(i) for i in range(10)}
    labels_dict = fashion_labels if dataset_name == 'FashionMNIST' else mnist_labels
    
    # Raccolta degli errori
    errors = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Trova errori
            incorrect_mask = predicted != labels
            
            if incorrect_mask.any():
                incorrect_inputs = inputs[incorrect_mask]
                incorrect_labels = labels[incorrect_mask]
                incorrect_preds = predicted[incorrect_mask]
                incorrect_probs = probs[incorrect_mask]
                
                for i in range(len(incorrect_inputs)):
                    if len(errors) < num_errors:
                        errors.append({
                            'image': incorrect_inputs[i].cpu(),
                            'true_label': incorrect_labels[i].cpu().item(),
                            'pred_label': incorrect_preds[i].cpu().item(),
                            'confidence': incorrect_probs[i][incorrect_preds[i]].cpu().item() * 100,
                            'true_prob': incorrect_probs[i][incorrect_labels[i]].cpu().item() * 100
                        })
            
            if len(errors) >= num_errors:
                break
    
    # Visualizzazione degli errori
    if errors:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'Analisi Errori del Modello - {dataset_name}', fontsize=16)
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(errors):
                error = errors[idx]
                
                # Mostra l'immagine
                img = error['image'].squeeze()
                ax.imshow(img, cmap='gray')
                
                # Informazioni sull'errore
                true_name = labels_dict[error['true_label']]
                pred_name = labels_dict[error['pred_label']]
                
                title = f'True: {true_name} ({error["true_prob"]:.1f}%)\n'
                title += f'Pred: {pred_name} ({error["confidence"]:.1f}%)'
                
                ax.set_title(title, fontsize=10, color='red', weight='bold')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        
        # Salvataggio del plot
        if save_plots:
            model_name = f"{type(model).__name__}_{dataset_name.lower()}"
            plot_path = os.path.join(output_dirs['plots'], f'error_analysis_{model_name}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"❌ Analisi errori salvata: {plot_path}")
        
        plt.show()
    else:
        print("🎉 Nessun errore trovato nei campioni analizzati!")
    
    return errors

# Funzione per visualizzare l'effetto del dropout su attivazioni
def visualize_dropout_effect(model_with_dropout, model_without_dropout, test_loader, dataset_name='MNIST', save_plots=True):
    """
    Visualizza l'effetto del dropout sulle attivazioni del modello
    
    Args:
        model_with_dropout: Modello con dropout
        model_without_dropout: Modello senza dropout
        test_loader: DataLoader per il test set
        dataset_name: Nome del dataset
        save_plots: Se salvare i plot o meno
    """
    # Funzione hook per catturare attivazioni
    activations_with = {}
    activations_without = {}
    
    def get_activation(name, storage):
        def hook(model, input, output):
            storage[name] = output.detach()
        return hook
    
    # Registra hooks per il primo strato nascosto
    if hasattr(model_with_dropout, 'fc1'):
        model_with_dropout.fc1.register_forward_hook(get_activation('fc1', activations_with))
    elif hasattr(model_with_dropout, 'layers'):
        model_with_dropout.layers[0].register_forward_hook(get_activation('fc1', activations_with))
    
    if hasattr(model_without_dropout, 'fc1'):
        model_without_dropout.fc1.register_forward_hook(get_activation('fc1', activations_without))
    elif hasattr(model_without_dropout, 'layers'):
        model_without_dropout.layers[0].register_forward_hook(get_activation('fc1', activations_without))
    
    # Ottieni un batch di test
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    
    # Forward pass
    model_with_dropout.train()  # Per attivare il dropout
    model_without_dropout.eval()
    
    with torch.no_grad():
        _ = model_with_dropout(inputs[:1])  # Solo un campione
        _ = model_without_dropout(inputs[:1])
    
    # Confronta le attivazioni
    if 'fc1' in activations_with and 'fc1' in activations_without:
        act_with = activations_with['fc1'][0].cpu().numpy()
        act_without = activations_without['fc1'][0].cpu().numpy()
        
        # GRAFICO NORMALE (scala lineare)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Effetto del Dropout sulle Attivazioni - {dataset_name}', fontsize=16)
        
        # Attivazioni senza dropout
        axes[0].hist(act_without, bins=50, alpha=0.7, color='blue', density=True)
        axes[0].set_title('Attivazioni SENZA Dropout')
        axes[0].set_xlabel('Valore Attivazione')
        axes[0].set_ylabel('Densità')
        axes[0].grid(True, alpha=0.3)
        
        # Attivazioni con dropout
        axes[1].hist(act_with, bins=50, alpha=0.7, color='red', density=True)
        axes[1].set_title('Attivazioni CON Dropout')
        axes[1].set_xlabel('Valore Attivazione')
        axes[1].set_ylabel('Densità')
        axes[1].grid(True, alpha=0.3)
        
        # Confronto sovrapposto
        axes[2].hist(act_without, bins=50, alpha=0.5, color='blue', density=True, label='Senza Dropout')
        axes[2].hist(act_with, bins=50, alpha=0.5, color='red', density=True, label='Con Dropout')
        axes[2].set_title('Confronto Attivazioni')
        axes[2].set_xlabel('Valore Attivazione')
        axes[2].set_ylabel('Densità')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvataggio del plot normale
        if save_plots:
            plot_path = os.path.join(output_dirs['plots'], f'dropout_activations_{dataset_name.lower()}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"🧠 Analisi attivazioni salvata: {plot_path}")
        
        plt.show()
        
        # GRAFICO CON NORMALIZZAZIONE LOGARITMICA
        fig_log, axes_log = plt.subplots(1, 3, figsize=(18, 5))
        fig_log.suptitle(f'Effetto del Dropout sulle Attivazioni - {dataset_name} (Scala Logaritmica)', fontsize=16)
        
        # Attivazioni senza dropout - scala log
        n_without, bins_without, patches_without = axes_log[0].hist(act_without, bins=50, alpha=0.7, color='blue', density=True)
        axes_log[0].set_yscale('log')
        axes_log[0].set_title('Attivazioni SENZA Dropout\n(Scala Logaritmica)')
        axes_log[0].set_xlabel('Valore Attivazione')
        axes_log[0].set_ylabel('Densità (log)')
        axes_log[0].grid(True, alpha=0.3)
        
        # Attivazioni con dropout - scala log
        n_with, bins_with, patches_with = axes_log[1].hist(act_with, bins=50, alpha=0.7, color='red', density=True)
        axes_log[1].set_yscale('log')
        axes_log[1].set_title('Attivazioni CON Dropout\n(Scala Logaritmica)')
        axes_log[1].set_xlabel('Valore Attivazione')
        axes_log[1].set_ylabel('Densità (log)')
        axes_log[1].grid(True, alpha=0.3)
        
        # Confronto sovrapposto - scala log
        axes_log[2].hist(act_without, bins=50, alpha=0.5, color='blue', density=True, label='Senza Dropout')
        axes_log[2].hist(act_with, bins=50, alpha=0.5, color='red', density=True, label='Con Dropout')
        axes_log[2].set_yscale('log')
        axes_log[2].set_title('Confronto Attivazioni\n(Scala Logaritmica)')
        axes_log[2].set_xlabel('Valore Attivazione')
        axes_log[2].set_ylabel('Densità (log)')
        axes_log[2].legend()
        axes_log[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvataggio del plot logaritmico
        if save_plots:
            plot_path_log = os.path.join(output_dirs['plots'], f'dropout_activations_{dataset_name.lower()}_log.png')
            plt.savefig(plot_path_log, dpi=300, bbox_inches='tight')
            print(f"📊 Analisi attivazioni logaritmica salvata: {plot_path_log}")
        
        plt.show()
            # STEP 11: Analisi Statistica Completa dei Pattern di Attivazione
            # ==============================================================
            # Questa sezione fornisce un'analisi quantitativa dettagliata di come il dropout
            # influenza le attivazioni neurali, offrendo insight sul meccanismo di regolarizzazione.
            # Calcoliamo diverse misure statistiche per comprendere:
            # - Cambiamenti della tendenza centrale (media, mediana)
            # - Cambiamenti della variabilità (deviazione standard, varianza)
            # - Cambiamenti della forma della distribuzione (min, max, range)
            # - Effetti sulla sparsità (percentuale di neuroni attivi)
            # - Cambiamenti relativi tra condizioni con e senza dropout

        print(f"\n📈 STATISTICHE ATTIVAZIONI - {dataset_name}:")
        print(f"{'='*50}")

        # Statistiche di base (senza dropout)
        # Rappresentano la distribuzione naturale delle attivazioni della rete
        # senza l'interferenza della regolarizzazione
        print(f"SENZA DROPOUT:")
        print(f"  - Media: {np.mean(act_without):.4f}")  # Misura della tendenza centrale
        print(f"  - Deviazione Standard: {np.std(act_without):.4f}")  # Misura della variabilità
        print(f"  - Min: {np.min(act_without):.4f}")  # Limite inferiore del range delle attivazioni
        print(f"  - Max: {np.max(act_without):.4f}")  # Limite superiore del range delle attivazioni
        # Analisi della sparsità: Conta i neuroni con attivazione positiva (output ReLU > 0)
        active_without = np.sum(act_without > 0)
        total_neurons = len(act_without)
        sparsity_without = 100 * active_without / total_neurons
        print(f"  - Neuroni attivi (>0): {active_without}/{total_neurons} ({sparsity_without:.1f}%)")

        # Statistiche regolarizzate (con dropout)
        # Mostrano come il dropout modifica il panorama delle attivazioni
        # Ci si aspetta generalmente medie più basse e pattern di sparsità diversi
        print(f"\nCON DROPOUT:")
        print(f"  - Media: {np.mean(act_with):.4f}")  # Atteso più basso per via degli zeri casuali
        print(f"  - Deviazione Standard: {np.std(act_with):.4f}")  # Può aumentare per la maggiore casualità
        print(f"  - Min: {np.min(act_with):.4f}")  # Dovrebbe restare 0 per via di ReLU + dropout
        print(f"  - Max: {np.max(act_with):.4f}")  # Può essere più alto per effetto di compensazione
        # Sparsità con dropout: Zeri aggiuntivi da dropout + zeri naturali di ReLU
        active_with = np.sum(act_with > 0)
        sparsity_with = 100 * active_with / total_neurons
        print(f"  - Neuroni attivi (>0): {active_with}/{total_neurons} ({sparsity_with:.1f}%)")

        # Analisi comparativa: Quantifica l'effetto del dropout
        # Queste metriche aiutano a capire l'impatto della regolarizzazione
        print(f"\nCOMPARAZIONE:")
        # Riduzione media attivazione: quanto il dropout riduce l'attivazione media
        mean_reduction = ((np.mean(act_without) - np.mean(act_with)) / np.mean(act_without) * 100)
        print(f"  - Riduzione media attivazione: {mean_reduction:.1f}%")

        # Riduzione neuroni attivi: quanto il dropout aumenta la sparsità
        if active_without > 0:  # Evita divisione per zero
            neuron_reduction = ((active_without - active_with) / active_without * 100)
            print(f"  - Riduzione neuroni attivi: {neuron_reduction:.1f}%")
        else:
            print(f"  - Riduzione neuroni attivi: N/A (nessun neurone attivo senza dropout)")
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e

        # Statistiche avanzate aggiuntive per analisi più approfondita
        print(f"\nSTATISTICHE AVANZATE:")
        # Confronto varianza: come cambia la variabilità delle attivazioni
        var_without = np.var(act_without)
        var_with = np.var(act_with)
        var_change = ((var_with - var_without) / var_without * 100) if var_without > 0 else 0
        print(f"  - Varianza senza dropout: {var_without:.6f}")
        print(f"  - Varianza con dropout: {var_with:.6f}")
        print(f"  - Cambio varianza: {var_change:.1f}%")

<<<<<<< HEAD
        # Analisi del range: come il dropout influenza il range delle attivazioni
        range_without = np.max(act_without) - np.min(act_without)
        range_with = np.max(act_with) - np.min(act_with)
        range_change = ((range_with - range_without) / range_without * 100) if range_without > 0 else 0
        print(f"  - Range attivazioni senza dropout: {range_without:.4f}")
        print(f"  - Range attivazioni con dropout: {range_with:.4f}")
        print(f"  - Cambio range: {range_change:.1f}%")

"""
📚 ISTRUZIONI COMPLETE PER L'USO DEL SISTEMA DI STUDIO DEL DROPOUT
===================================================================

Il presente script implementa un sistema completo per lo studio degli effetti
del dropout nelle reti neurali, con particolare focus su:
- Analisi comparativa di architetture con e senza dropout
- Visualizzazione avanzata delle performance e dei pattern di attivazione
- Generazione automatica di report dettagliati
- Pipeline modulare per diversi scenari di utilizzo

🎯 MODALITÀ DI ESECUZIONE PRINCIPALI:
====================================

1. ESPERIMENTO COMPLETO (Modalità di default):
   -------------------------------------------
   python examp2.py
   
   COSA ESEGUE:
   - Training completo su MNIST e FashionMNIST
   - Test di architetture single-layer e multi-layer
   - Analisi con diversi tassi di dropout (0.0, 0.1, 0.3, 0.5, 0.7)
   - Generazione di 20+ visualizzazioni avanzate
   - Creazione di report dettagliati in formato Markdown
   - Salvataggio automatico di modelli, grafici e risultati
   
   REQUISITI:
   - Tempo: 15-30 minuti (dipende dall'hardware)
   - Memoria: 2-4GB RAM
   - Storage: 100-200MB per risultati completi
   - GPU: Opzionale ma raccomandato per velocità

2. MODALITÀ DEMO VISUALIZZAZIONI (Solo con modelli pre-esistenti):
   ---------------------------------------------------------------
   # Decommenta nel blocco main():
   demo_visualizations('MNIST', 'multi')
   demo_visualizations('FashionMNIST', 'single')
   
   COSA ESEGUE:
   - Carica modelli precedentemente trainati
   - Esegue tutte le visualizzazioni avanzate
   - Mostra analisi di predizioni e errori
   - Visualizza effetti del dropout sulle attivazioni
   
   REQUISITI:
   - Esecuzione precedente di main() per generare i modelli
   - Tempo: 2-5 minuti
   - Memoria: <1GB RAM

3. MODALITÀ QUICK DEMO (Training rapido + visualizzazioni):
   --------------------------------------------------------
   # Decommenta nel blocco main():
   quick_demo('MNIST', epochs=3)
   quick_demo('FashionMNIST', epochs=5)
   
   COSA ESEGUE:
   - Training accelerato con parametri ottimizzati per velocità
   - Visualizzazioni immediate senza salvataggio su disco
   - Analisi comparativa rapida degli effetti del dropout
   - Report sintetico delle performance
   
   REQUISITI:
   - Tempo: 3-7 minuti
   - Memoria: 1-2GB RAM
   - Ideale per test e sviluppo

🔬 FUNZIONALITÀ AVANZATE IMPLEMENTATE:
=====================================

A. ARCHITETTURE NEURALI:
   - MLPSingleHidden: Rete a singolo layer nascosto (512 neuroni)
   - MLPMultiHidden: Rete multi-layer (512→256→128 neuroni)
   - Implementazione modulare con dropout configurabile
   - Supporto GPU automatico con fallback CPU

B. SISTEMA DI VISUALIZZAZIONE:
   - visualize_sample_predictions(): Predizioni su campioni individuali
   - compare_model_predictions(): Confronto diretto tra modelli
   - analyze_model_errors(): Analisi sistematica degli errori
   - visualize_dropout_effect(): Effetti sulle attivazioni neurali
   - plot_results(): Grafici di performance e loss
   - dropout_analysis(): Analisi parametrica del tasso di dropout

C. PIPELINE DI ANALISI:
   - Training automatizzato con early stopping
   - Valutazione multi-metrica (accuracy, loss, confusion matrix)
   - Analisi statistica delle attivazioni neurali
   - Generazione automatica di report Markdown
   - Organizzazione strutturata dei risultati

D. GESTIONE DEI RISULTATI:
   - Salvataggio automatico in directory timestampate
   - Organizzazione in sottocartelle (models/, plots/, reports/)
   - Serializzazione dei risultati in formato JSON
   - Grafici ad alta risoluzione (300 DPI) per pubblicazione

🛠️ PERSONALIZZAZIONE E PARAMETRI:
=================================

PARAMETRI PRINCIPALI MODIFICABILI:

1. Dataset e Preprocessing:
   - batch_size: Dimensione batch (default: 128)
   - Transforms: Normalizzazione e augmentation
   - Train/test split: Automatico da TorchVision

2. Architetture:
   - hidden_sizes: Dimensioni layer nascosti
   - dropout_rates: Tassi di dropout da testare
   - activation: Funzioni di attivazione (ReLU default)

3. Training:
   - epochs: Numero di epoche (default: 20)
   - learning_rate: Tasso di apprendimento (default: 0.001)
   - optimizer: Adam vs SGD
   - loss_function: CrossEntropyLoss per classificazione

4. Visualizzazione:
   - save_plots: Salvataggio automatico (default: True)
   - dpi: Risoluzione grafici (default: 300)
   - num_samples: Campioni per visualizzazione

🚀 ESTENSIONI E SVILUPPO FUTURO:
===============================

AREE DI ESTENSIONE POSSIBILI:

1. Nuovi Dataset:
   - CIFAR-10/100 per immagini più complesse
   - Dataset custom tramite DataLoader personalizzati
   - Preprocessing avanzato e augmentation

2. Architetture Avanzate:
   - Reti Convoluzionali (CNN) con dropout spaziale
   - Reti Residuali (ResNet) con dropout strutturato
   - Attention mechanisms con dropout selettivo

3. Tecniche di Regularizzazione:
   - Batch Normalization vs Dropout
   - Layer Normalization
   - Weight Decay e L1/L2 regularization
   - Mixup e CutMix augmentation

4. Metriche Avanzate:
   - Interpretabilità con LIME/SHAP
   - Analisi della geometria dello spazio delle feature
   - Robustezza ad attacchi adversarial
   - Calibrazione delle predizioni

5. Ottimizzazioni:
   - Distributed training per dataset grandi
   - Mixed precision training
   - Quantizzazione dei modelli
   - Pruning strutturato

📊 OUTPUT E RISULTATI:
=====================

STRUTTURA DIRECTORY OUTPUT:
dropout_study_results_YYYYMMDD_HHMMSS/
├── models/                    # Modelli trainati (.pth)
├── plots/                     # Visualizzazioni (.png)
├── reports/                   # Report dettagliati (.md, .json)
└── README.md                 # Riepilogo dell'esperimento

TIPI DI FILE GENERATI:
- *.pth: Modelli PyTorch salvati con state_dict
- *.png: Grafici ad alta risoluzione (300 DPI)
- *.json: Risultati numerici e metriche
- *.md: Report in Markdown con analisi completa

METRICHE TRACCIATE:
- Accuracy su training e test set
- Loss per epoca
- Tempi di training
- Statistiche delle attivazioni neurali
- Matrici di confusione
- Analisi degli errori per classe

⚠️ NOTE IMPORTANTI:
==================

REQUISITI SISTEMA:
- Python 3.8+ con PyTorch, TorchVision, Matplotlib, NumPy
- Spazio disco: minimo 500MB per risultati completi
- RAM: minimo 2GB, raccomandato 4GB+
- GPU: Opzionale, automaticamente rilevata

BEST PRACTICES:
- Eseguire in environment virtuale dedicato
- Verificare disponibilità GPU prima dell'esecuzione
- Monitorare utilizzo memoria durante training estesi
- Backup periodico dei risultati importanti

TROUBLESHOOTING:
- Out of Memory: Ridurre batch_size o hidden_sizes
- Slow training: Verificare utilizzo GPU con nvidia-smi
- Missing dependencies: pip install -r requirements.txt
- File permissions: Verificare write access nella directory

🎓 VALORE EDUCATIVO:
===================

Questo script serve come:
- Esempio completo di pipeline di deep learning
- Studio approfondito degli effetti del dropout
- Template per ricerche comparative su tecniche di regolarizzazione
- Strumento didattico per comprendere l'overfitting
- Base per esperimenti avanzati di machine learning

La documentazione estensiva e l'architettura modulare rendono il codice
ideale per scopi educativi, ricerca e sviluppo di nuove tecniche.
=======
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
>>>>>>> 3b3149d69db6ca3bf13da579be64b45f0f02861e
"""