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

# Impostazione del device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilizzo device: {device}")

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

# Funzione per confrontare modelli
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
    
    return results

# Funzione per visualizzare i risultati
def plot_results(results, dataset_name):
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
    plt.show()

# Funzione per analisi dettagliata del dropout
def dropout_analysis(dataset_name='MNIST'):
    print(f"\n=== ANALISI DETTAGLIATA DROPOUT SU {dataset_name} ===")
    
    train_loader, test_loader = load_datasets(dataset_name)
    
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}
    
    input_size = 28 * 28
    num_classes = 10
    num_epochs = 10
    learning_rate = 0.001
    
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
        
        print(f"Train Acc: {train_acc[-1]:.2f}%, Test Acc: {test_acc:.2f}%, Gap: {train_acc[-1] - test_acc:.2f}%")
    
    # Visualizzazione risultati dropout analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
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
    plt.show()
    
    return results

# Funzione per stampare report finale
def print_summary_report(results_mnist, results_fashion):
    print("\n" + "="*80)
    print("REPORT FINALE - STUDIO DEL DROPOUT")
    print("="*80)
    
    print("\nMNIST Dataset:")
    print("-" * 50)
    for model_name, data in results_mnist.items():
        print(f"{model_name:20} | Test Acc: {data['test_acc']:6.2f}% | "
              f"Train-Test Gap: {data['train_acc'][-1] - data['test_acc']:5.2f}% | "
              f"Time: {data['training_time']:5.1f}s")
    
    print("\nFashion-MNIST Dataset:")
    print("-" * 50)
    for model_name, data in results_fashion.items():
        print(f"{model_name:20} | Test Acc: {data['test_acc']:6.2f}% | "
              f"Train-Test Gap: {data['train_acc'][-1] - data['test_acc']:5.2f}% | "
              f"Time: {data['training_time']:5.1f}s")
    
    print("\nCONCLUSIONI:")
    print("-" * 50)
    print("• Il dropout riduce l'overfitting (gap train-test minore)")
    print("• L'effetto è più pronunciato su reti più profonde")
    print("• Fashion-MNIST è più difficile di MNIST (accuratezze inferiori)")
    print("• Il dropout può leggermente ridurre l'accuratezza finale ma migliora la generalizzazione")

# Funzione principale
def main():
    print("STUDIO DEL DROPOUT COME TECNICA DI REGOLARIZZAZIONE")
    print("="*60)
    
    # Confronto su MNIST
    results_mnist = compare_models('MNIST')
    plot_results(results_mnist, 'MNIST')
    
    # Confronto su Fashion-MNIST
    results_fashion = compare_models('FashionMNIST')
    plot_results(results_fashion, 'Fashion-MNIST')
    
    # Analisi dettagliata dropout
    print("\nAnalisi dettagliata dell'effetto del dropout...")
    dropout_results_mnist = dropout_analysis('MNIST')
    dropout_results_fashion = dropout_analysis('FashionMNIST')
    
    # Report finale
    print_summary_report(results_mnist, results_fashion)

if __name__ == "__main__":
    main()