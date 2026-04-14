import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from modules import model
from modules.model import AMRModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    SNRS = np.array(SNRS)

    X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
        X, y, SNRS, test_size=1-train_split, stratify=y, random_state=42
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, le.classes_, X_test, y_test, snr_test, SNRS

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

def evaluate_snr(model, X_test, y_test, snr_test, device, epoch):
    model.eval()

    snr_correct = defaultdict(int)
    snr_total = defaultdict(int)

    all_preds = []
    all_targets = []
    all_snrs = []

    with torch.no_grad():
        for i in range(0, len(X_test), 256):
            x = X_test[i:i+256].to(device)
            y = y_test[i:i+256].to(device)
            snr_batch = snr_test[i:i+256]

            outputs = model(x)
            _, preds = outputs.max(1)

            preds = preds.cpu().numpy()
            y_np = y.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(y_np)
            all_snrs.extend(snr_batch)

            for p, t, s in zip(preds, y_np, snr_batch):
                snr_total[s] += 1
                if p == t:
                    snr_correct[s] += 1

    # Compute accuracy per SNR
    snrs_sorted = sorted(snr_total.keys())
    accs = [100 * snr_correct[s] / snr_total[s] for s in snrs_sorted]

    # Plot
    plt.figure()
    plt.plot(snrs_sorted, accs, marker='o')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"SNR vs Accuracy (Epoch {epoch})")
    plt.grid()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/snr_vs_acc_epoch_{epoch}.png")
    plt.close()

    return snrs_sorted, accs, all_preds, all_targets, all_snrs

def evaluate_18db(all_preds, all_targets, all_snrs, class_names):
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    snrs = np.array(all_snrs)

    mask = snrs == 18

    preds_18 = preds[mask]
    targets_18 = targets[mask]

    acc = (preds_18 == targets_18).mean() * 100
    print(f"\n🔥 18 dB Accuracy: {acc:.2f}%")

    cm = confusion_matrix(targets_18, preds_18)

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap='Blues', values_format='d')

    with open("plots/18db_accuracy.txt", "w") as f:
        f.write(f"{acc:.4f}")

    plt.title("Confusion Matrix (18 dB)")
    plt.savefig("plots/conf_matrix_18db.png")
    plt.close()

def main():
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

    if not os.path.exists('plots'):
        os.makedirs('plots', exist_ok=True)

    # Load data
    print("Loading data...")
    train_loader, test_loader, class_names, X_test, y_test, snr_test, SNRS = load_data(r'data/RML2016.10a_dict.pkl')

    # SAVE LABEL ENCODER
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(class_names, f)   # store class names instead (simpler)

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

        snrs_sorted, accs, all_preds, all_targets, all_snrs = evaluate_snr(
            model, X_test, y_test, snr_test, DEVICE, epoch+1
        )
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print("✓ Model saved!")
    
    print(f"\n=== Training Complete ===")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")

    # LOAD BEST MODEL AGAIN
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()

    # Recompute predictions using BEST model
    _, _, all_preds, all_targets, all_snrs = evaluate_snr(
        model, X_test, y_test, snr_test, DEVICE, "best"
    )

    evaluate_18db(all_preds, all_targets, all_snrs, class_names)
        
    return model, train_losses, train_accs, test_losses, test_accs

if __name__ == "__main__":
    model, train_losses, train_accs, test_losses, test_accs = main()