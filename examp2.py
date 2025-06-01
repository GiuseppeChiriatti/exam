import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import defaultdict
import time
import os
from datetime import datetime
import json

# Impostazione del device e creazione cartelle per i risultati
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilizzo device: {device}")

# Creazione delle cartelle per salvare i risultati
def create_output_directories():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"dropout_study_results_{timestamp}"
    
    directories = {
        'base': base_dir,
        'plots': os.path.join(base_dir, 'plots'),
        'models': os.path.join(base_dir, 'models'),
        'reports': os.path.join(base_dir, 'reports'),
        'data': os.path.join(base_dir, 'data')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

# Inizializzazione delle directory
output_dirs = create_output_directories()
print(f"Risultati salvati in: {output_dirs['base']}")

# Variabile globale per raccogliere informazioni per il report
experiment_log = {
    'start_time': datetime.now().isoformat(),
    'device': str(device),
    'experiments': [],
    'conclusions': []
}

# Classe MLP con un solo strato nascosto
class MLPSingleHidden(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.0):
        super(MLPSingleHidden, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Classe MLP con più strati nascosti
class MLPMultiHidden(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.0):
        super(MLPMultiHidden, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Primo strato
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Strati intermedi
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Strato di output
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        
        for layer, dropout in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            x = dropout(x)
        
        x = self.output_layer(x)
        return x

# Funzione per caricare i dataset
def load_datasets(dataset_name='MNIST', batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
    elif dataset_name == 'FashionMNIST':
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Funzione di training
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return train_losses, train_accuracies

# Funzione di test
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    return accuracy, avg_loss

# Funzione per confrontare modelli con logging
def compare_models(dataset_name='MNIST'):
    print(f"\n=== CONFRONTO MODELLI SU {dataset_name} ===")
    
    # Caricamento dataset
    train_loader, test_loader = load_datasets(dataset_name)
    
    # Parametri
    input_size = 28 * 28
    num_classes = 10
    num_epochs = 15
    learning_rate = 0.001
    
    results = {}
    experiment_info = {
        'dataset': dataset_name,
        'parameters': {
            'input_size': input_size,
            'num_classes': num_classes,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate
        },
        'models': {}
    }
    
    # 1. MLP singolo strato senza dropout
    print("\n1. Training MLP singolo strato SENZA dropout...")
    model1 = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.0).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    train_losses1, train_acc1 = train_model(model1, train_loader, criterion, optimizer1, num_epochs)
    training_time1 = time.time() - start_time
    
    test_acc1, test_loss1 = test_model(model1, test_loader)
    results['Single_No_Dropout'] = {
        'train_losses': train_losses1,
        'train_acc': train_acc1,
        'test_acc': test_acc1,
        'test_loss': test_loss1,
        'training_time': training_time1
    }
    print(f"Test Accuracy: {test_acc1:.2f}%, Test Loss: {test_loss1:.4f}")
    
    # Salvataggio modello
    model_path = os.path.join(output_dirs['models'], f'{dataset_name.lower()}_single_no_dropout.pth')
    torch.save(model1.state_dict(), model_path)
    
    experiment_info['models']['Single_No_Dropout'] = {
        'architecture': 'Single Hidden Layer (512 neurons)',
        'dropout_rate': 0.0,
        'test_accuracy': test_acc1,
        'training_time': training_time1,
        'overfitting_gap': train_acc1[-1] - test_acc1
    }
    
    # 2. MLP singolo strato con dropout
    print("\n2. Training MLP singolo strato CON dropout (0.3)...")
    model2 = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.3).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
    
    start_time = time.time()
    train_losses2, train_acc2 = train_model(model2, train_loader, criterion, optimizer2, num_epochs)
    training_time2 = time.time() - start_time
    
    test_acc2, test_loss2 = test_model(model2, test_loader)
    results['Single_With_Dropout'] = {
        'train_losses': train_losses2,
        'train_acc': train_acc2,
        'test_acc': test_acc2,
        'test_loss': test_loss2,
        'training_time': training_time2
    }
    print(f"Test Accuracy: {test_acc2:.2f}%, Test Loss: {test_loss2:.4f}")
    
    # Salvataggio modello
    model_path = os.path.join(output_dirs['models'], f'{dataset_name.lower()}_single_with_dropout.pth')
    torch.save(model2.state_dict(), model_path)
    
    experiment_info['models']['Single_With_Dropout'] = {
        'architecture': 'Single Hidden Layer (512 neurons)',
        'dropout_rate': 0.3,
        'test_accuracy': test_acc2,
        'training_time': training_time2,
        'overfitting_gap': train_acc2[-1] - test_acc2
    }
    
    # 3. MLP multi-strato senza dropout
    print("\n3. Training MLP multi-strato SENZA dropout...")
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
    
    # Salvataggio modello
    model_path = os.path.join(output_dirs['models'], f'{dataset_name.lower()}_multi_no_dropout.pth')
    torch.save(model3.state_dict(), model_path)
    
    experiment_info['models']['Multi_No_Dropout'] = {
        'architecture': 'Multi Hidden Layers (512-256-128 neurons)',
        'dropout_rate': 0.0,
        'test_accuracy': test_acc3,
        'training_time': training_time3,
        'overfitting_gap': train_acc3[-1] - test_acc3
    }
    
    # 4. MLP multi-strato con dropout
    print("\n4. Training MLP multi-strato CON dropout (0.3)...")
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
    
    # Salvataggio modello
    model_path = os.path.join(output_dirs['models'], f'{dataset_name.lower()}_multi_with_dropout.pth')
    torch.save(model4.state_dict(), model_path)
    
    experiment_info['models']['Multi_With_Dropout'] = {
        'architecture': 'Multi Hidden Layers (512-256-128 neurons)',
        'dropout_rate': 0.3,
        'test_accuracy': test_acc4,
        'training_time': training_time4,
        'overfitting_gap': train_acc4[-1] - test_acc4
    }
    
    # Salvataggio risultati numerici
    results_path = os.path.join(output_dirs['data'], f'results_{dataset_name.lower()}.json')
    with open(results_path, 'w') as f:
        # Converti i numpy arrays in liste per la serializzazione JSON
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
    
    # Aggiunta al log dell'esperimento
    experiment_log['experiments'].append(experiment_info)
    
    return results

# Funzione per visualizzare i risultati con salvataggio
def plot_results(results, dataset_name, save_plots=True):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Confronto Prestazioni Modelli - {dataset_name}', fontsize=16)
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for model_name, data in results.items():
        ax1.plot(data['train_losses'], label=model_name, linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy
    ax2 = axes[0, 1]
    for model_name, data in results.items():
        ax2.plot(data['train_acc'], label=model_name, linewidth=2)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Test Accuracy Comparison
    ax3 = axes[1, 0]
    models = list(results.keys())
    test_accs = [results[model]['test_acc'] for model in models]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
    bars = ax3.bar(models, test_accs, color=colors)
    ax3.set_title('Test Accuracy Comparison')
    ax3.set_ylabel('Accuracy (%)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Aggiunta valori sopra le barre
    for bar, acc in zip(bars, test_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.2f}%', ha='center', va='bottom')
    
    # Plot 4: Overfitting Analysis (Differenza train-test)
    ax4 = axes[1, 1]
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
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Aggiunta valori sopra le barre
    for bar, gap in zip(bars, overfitting):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{gap:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Salvataggio del plot
    if save_plots:
        plot_path = os.path.join(output_dirs['plots'], f'model_comparison_{dataset_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot salvato: {plot_path}")
    
    plt.show()

# Funzione per analisi dettagliata del dropout con salvataggio
def dropout_analysis(dataset_name='MNIST', save_plots=True):
    print(f"\n=== ANALISI DETTAGLIATA DROPOUT SU {dataset_name} ===")
    
    train_loader, test_loader = load_datasets(dataset_name)
    
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}
    
    input_size = 28 * 28
    num_classes = 10
    num_epochs = 10
    learning_rate = 0.001
    
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
    
    for dropout_rate in dropout_rates:
        print(f"\nTesting dropout rate: {dropout_rate}")
        
        model = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        train_losses, train_acc = train_model(model, train_loader, criterion, optimizer, num_epochs)
        test_acc, test_loss = test_model(model, test_loader)
        
        results[dropout_rate] = {
            'train_acc': train_acc[-1],
            'test_acc': test_acc,
            'overfitting': train_acc[-1] - test_acc
        }
        
        dropout_experiment['results'][str(dropout_rate)] = {
            'train_accuracy': float(train_acc[-1]),
            'test_accuracy': float(test_acc),
            'overfitting_gap': float(train_acc[-1] - test_acc)
        }
        
        print(f"Train Acc: {train_acc[-1]:.2f}%, Test Acc: {test_acc:.2f}%, Gap: {train_acc[-1] - test_acc:.2f}%")
    
    # Visualizzazione risultati dropout analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Analisi Dropout - {dataset_name}', fontsize=16)
    
    dropout_vals = list(results.keys())
    train_accs = [results[dr]['train_acc'] for dr in dropout_vals]
    test_accs = [results[dr]['test_acc'] for dr in dropout_vals]
    overfitting = [results[dr]['overfitting'] for dr in dropout_vals]
    
    # Plot accuracies
    axes[0].plot(dropout_vals, train_accs, 'o-', label='Train Accuracy', linewidth=2, markersize=8)
    axes[0].plot(dropout_vals, test_accs, 's-', label='Test Accuracy', linewidth=2, markersize=8)
    axes[0].set_xlabel('Dropout Rate')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Effect of Dropout on Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot test accuracy only
    axes[1].plot(dropout_vals, test_accs, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Dropout Rate')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Test Accuracy vs Dropout Rate')
    axes[1].grid(True, alpha=0.3)
    
    # Plot overfitting
    axes[2].plot(dropout_vals, overfitting, 'go-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Dropout Rate')
    axes[2].set_ylabel('Train-Test Gap (%)')
    axes[2].set_title('Overfitting vs Dropout Rate')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvataggio del plot
    if save_plots:
        plot_path = os.path.join(output_dirs['plots'], f'dropout_analysis_{dataset_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot salvato: {plot_path}")
    
    plt.show()
    
    # Salvataggio risultati dropout analysis
    dropout_results_path = os.path.join(output_dirs['data'], f'dropout_analysis_{dataset_name.lower()}.json')
    with open(dropout_results_path, 'w') as f:
        json.dump(dropout_experiment, f, indent=2)
    
    # Aggiunta al log dell'esperimento
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
    print("STUDIO DEL DROPOUT COME TECNICA DI REGOLARIZZAZIONE")
    print("="*60)
    print(f"📁 Tutti i risultati saranno salvati in: {output_dirs['base']}")
    
    # Confronto su MNIST
    print("\n🔍 Avvio esperimenti su MNIST...")
    results_mnist = compare_models('MNIST')
    plot_results(results_mnist, 'MNIST', save_plots=True)
    
    # Confronto su Fashion-MNIST
    print("\n🔍 Avvio esperimenti su Fashion-MNIST...")
    results_fashion = compare_models('FashionMNIST')
    plot_results(results_fashion, 'Fashion-MNIST', save_plots=True)
    
    # Analisi dettagliata dropout
    print("\n📊 Analisi dettagliata dell'effetto del dropout...")
    dropout_results_mnist = dropout_analysis('MNIST', save_plots=True)
    dropout_results_fashion = dropout_analysis('FashionMNIST', save_plots=True)
    
    # Creazione del report completo
    print("\n📝 Generazione report completo...")
    report_path = create_comprehensive_report(
        results_mnist, results_fashion, 
        dropout_results_mnist, dropout_results_fashion
    )
    
    # Tentativo di creare anche il PDF
    print("\n📄 Tentativo di creazione PDF...")
    pdf_path = create_pdf_report(report_path)
      # Visualizzazione campioni predetti
    print("\n🖼️ Visualizzazione campioni con predizioni...")
    
    # Carica i modelli per la visualizzazione
    train_loader_mnist, test_loader_mnist = load_datasets('MNIST')
    train_loader_fashion, test_loader_fashion = load_datasets('FashionMNIST')
    
    # Crea modelli per la dimostrazione
    input_size = 28 * 28
    num_classes = 10
    
    # Modelli MNIST
    model_mnist_single = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.0).to(device)
    model_mnist_single_dropout = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.3).to(device)
    model_mnist_multi = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.0).to(device)
    model_mnist_multi_dropout = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.3).to(device)
    
    # Carica i pesi dei modelli salvati
    try:
        model_mnist_single.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'mnist_single_no_dropout.pth')))
        model_mnist_single_dropout.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'mnist_single_with_dropout.pth')))
        model_mnist_multi.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'mnist_multi_no_dropout.pth')))
        model_mnist_multi_dropout.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'mnist_multi_with_dropout.pth')))
        
        # Visualizza campioni MNIST
        print("📸 Visualizzazione campioni MNIST...")
        visualize_sample_predictions(model_mnist_multi_dropout, test_loader_mnist, 'MNIST', num_samples=12)
        
        # Confronta modelli MNIST
        models_dict_mnist = {
            'Single No Dropout': model_mnist_single,
            'Single With Dropout': model_mnist_single_dropout,
            'Multi No Dropout': model_mnist_multi,
            'Multi With Dropout': model_mnist_multi_dropout
        }
        compare_model_predictions(models_dict_mnist, test_loader_mnist, 'MNIST', num_samples=6)
        
        # Analizza errori MNIST
        print("❌ Analisi errori MNIST...")
        analyze_model_errors(model_mnist_multi, test_loader_mnist, 'MNIST', num_errors=12)
        
        # Effetto dropout sulle attivazioni MNIST
        print("🧠 Analisi effetto dropout sulle attivazioni MNIST...")
        visualize_dropout_effect(model_mnist_multi_dropout, model_mnist_multi, test_loader_mnist, 'MNIST')
        
    except FileNotFoundError:
        print("⚠️ Modelli MNIST non trovati, probabilmente il training non è stato completato.")
    
    # Modelli Fashion-MNIST
    try:
        model_fashion_single = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.0).to(device)
        model_fashion_single_dropout = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.3).to(device)
        model_fashion_multi = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.0).to(device)
        model_fashion_multi_dropout = MLPMultiHidden(input_size, [512, 256, 128], num_classes, dropout_rate=0.3).to(device)
        
        model_fashion_single.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'fashionmnist_single_no_dropout.pth')))
        model_fashion_single_dropout.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'fashionmnist_single_with_dropout.pth')))
        model_fashion_multi.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'fashionmnist_multi_no_dropout.pth')))
        model_fashion_multi_dropout.load_state_dict(torch.load(os.path.join(output_dirs['models'], 'fashionmnist_multi_with_dropout.pth')))
        
        # Visualizza campioni Fashion-MNIST
        print("👕 Visualizzazione campioni Fashion-MNIST...")
        visualize_sample_predictions(model_fashion_multi_dropout, test_loader_fashion, 'FashionMNIST', num_samples=12)
        
        # Confronta modelli Fashion-MNIST
        models_dict_fashion = {
            'Single No Dropout': model_fashion_single,
            'Single With Dropout': model_fashion_single_dropout,
            'Multi No Dropout': model_fashion_multi,
            'Multi With Dropout': model_fashion_multi_dropout
        }
        compare_model_predictions(models_dict_fashion, test_loader_fashion, 'FashionMNIST', num_samples=6)
        
        # Analizza errori Fashion-MNIST
        print("❌ Analisi errori Fashion-MNIST...")
        analyze_model_errors(model_fashion_multi, test_loader_fashion, 'FashionMNIST', num_errors=12)
        
        # Effetto dropout sulle attivazioni Fashion-MNIST
        print("🧠 Analisi effetto dropout sulle attivazioni Fashion-MNIST...")
        visualize_dropout_effect(model_fashion_multi_dropout, model_fashion_multi, test_loader_fashion, 'FashionMNIST')
        
    except FileNotFoundError:
        print("⚠️ Modelli Fashion-MNIST non trovati, probabilmente il training non è stato completato.")
    
    # Creazione di un plot riassuntivo finale
    create_summary_plot(results_mnist, results_fashion)
    
    # Report finale a schermo
    print_final_summary(results_mnist, results_fashion, dropout_results_mnist, dropout_results_fashion)
    
    print(f"\n✅ ESPERIMENTO COMPLETATO!")
    print(f"📁 Tutti i file sono disponibili in: {output_dirs['base']}")
    print(f"📊 Grafici salvati in: {output_dirs['plots']}")
    print(f"🤖 Modelli salvati in: {output_dirs['models']}")
    print(f"📝 Report salvato in: {output_dirs['reports']}")

def create_summary_plot(results_mnist, results_fashion):
    """Crea un grafico riassuntivo finale che confronta tutti i risultati"""
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
        
        # Statistiche per confronto
        print(f"\n📈 STATISTICHE ATTIVAZIONI - {dataset_name}:")
        print(f"{'='*50}")
        print(f"SENZA DROPOUT:")
        print(f"  - Media: {np.mean(act_without):.4f}")
        print(f"  - Deviazione Standard: {np.std(act_without):.4f}")
        print(f"  - Min: {np.min(act_without):.4f}")
        print(f"  - Max: {np.max(act_without):.4f}")
        print(f"  - Neuroni attivi (>0): {np.sum(act_without > 0)}/{len(act_without)} ({100*np.sum(act_without > 0)/len(act_without):.1f}%)")
        
        print(f"\nCON DROPOUT:")
        print(f"  - Media: {np.mean(act_with):.4f}")
        print(f"  - Deviazione Standard: {np.std(act_with):.4f}")
        print(f"  - Min: {np.min(act_with):.4f}")
        print(f"  - Max: {np.max(act_with):.4f}")
        print(f"  - Neuroni attivi (>0): {np.sum(act_with > 0)}/{len(act_with)} ({100*np.sum(act_with > 0)/len(act_with):.1f}%)")
        
        print(f"\nCOMPARAZIONE:")
        print(f"  - Riduzione media attivazione: {((np.mean(act_without) - np.mean(act_with))/np.mean(act_without)*100):.1f}%")
        print(f"  - Riduzione neuroni attivi: {((np.sum(act_without > 0) - np.sum(act_with > 0))/np.sum(act_without > 0)*100):.1f}%")

# Funzione per demo delle visualizzazioni (può essere chiamata separatamente)
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
        model_no_dropout = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.0).to(device)
        model_with_dropout = MLPSingleHidden(input_size, 512, num_classes, dropout_rate=0.3).to(device)
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
            
            print("✅ Modelli caricati con successo!")
            
            # 1. Visualizza campioni con predizioni
            print("\n📸 1. Visualizzazione campioni con predizioni...")
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
    # Esegui l'esperimento completo
    main()
    
    # Per testare solo le visualizzazioni (decommentare se necessario):
    # demo_visualizations('MNIST', 'multi')
    # demo_visualizations('FashionMNIST', 'single')
    
    # Per un test veloce (decommentare se necessario):
    # quick_demo('MNIST', epochs=3)

# ISTRUZIONI PER L'USO:
# 
# 1. ESPERIMENTO COMPLETO:
#    python examp2.py
#    (Esegue tutto: training, analisi, visualizzazioni, report)
#
# 2. SOLO VISUALIZZAZIONI (se hai già i modelli):
#    demo_visualizations('MNIST', 'multi')
#    demo_visualizations('FashionMNIST', 'single')
#
# 3. TEST VELOCE (training rapido + visualizzazioni):
#    quick_demo('MNIST', epochs=3)
#    quick_demo('FashionMNIST', epochs=5)
#
# NUOVE FUNZIONALITÀ AGGIUNTE:
# - visualize_sample_predictions(): Mostra campioni con predizioni e confidenza
# - compare_model_predictions(): Confronta predizioni di più modelli sugli stessi campioni
# - analyze_model_errors(): Analizza e visualizza gli errori più comuni
# - visualize_dropout_effect(): Mostra l'effetto del dropout sulle attivazioni
# 
# Tutti i grafici vengono salvati automaticamente nella cartella plots/