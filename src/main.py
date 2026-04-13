import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from modules.model import AMRModel
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_CLASSES = 10

print(f"Using device: {DEVICE}")

def load_data(filepath, train_split=0.8):
    with open(filepath, "rb") as f:
        data = pickle.load(f, encoding="bytes")  # ✅ fix

    SNRS, MODS, IQ = [], [], []

    for (mod, snr), signal in data.items():
        for sample in signal:
            IQ.append(sample)
            MODS.append(mod)
            SNRS.append(snr)

    IQ = np.array(IQ)
    MODS = np.array(MODS)

    le = LabelEncoder()
    y = le.fit_transform(MODS)

    X = torch.tensor(IQ, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # ✅ IMPORTANT: shuffle split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-train_split, stratify=y, random_state=42
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, le.classes_

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training', leave=False)):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx + 1}] Loss: {loss.item():.4f}')
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def test(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing', leave=False):
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, target)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
    
    test_loss = total_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    return test_loss, test_acc

def main():
    if not os.path.exists('models'):
        os.makedirs('models')

    # Load data
    print("Loading data...")
    train_loader, test_loader, class_names = load_data(r'data/RML2016.10a_dict.pkl')

    NUM_CLASSES = len(class_names)   # ✅ dynamic
    print("Num classes:", NUM_CLASSES)

    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Initialize model, loss, and optimizer
    model = AMRModel(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    
    # Training loop
    best_test_acc = 0
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in tqdm(range(NUM_EPOCHS), desc='Epochs'):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Test
        test_loss, test_acc = test(model, test_loader, criterion, DEVICE)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print("✓ Model saved!")
    
    print(f"\n=== Training Complete ===")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    
    return model, train_losses, train_accs, test_losses, test_accs

if __name__ == "__main__":
    model, train_losses, train_accs, test_losses, test_accs = main()