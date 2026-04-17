import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from modules.model import AMRModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import csv

# ================= CONFIG =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

print(f"Using device: {DEVICE}")

# ================= DATA =================
def load_data(filepath, train_split=0.8):
    with open(filepath, "rb") as f:
        data = pickle.load(f, encoding="bytes")

    SNRS, MODS, IQ = [], [], []

    for (mod, snr), signal in data.items():
        for sample in signal:
            IQ.append(sample)
            MODS.append(mod)
            SNRS.append(snr)

    IQ = np.array(IQ)
    MODS = np.array(MODS)
    SNRS = np.array(SNRS)

    le = LabelEncoder()
    y = le.fit_transform(MODS)

    X = torch.tensor(IQ, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
        X, y, SNRS, test_size=1-train_split, stratify=y, random_state=42
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, le.classes_, X_test, y_test, snr_test

# ================= TRAIN =================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), 100 * correct / total

def test(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), 100 * correct / total

# ================= EVAL =================
def evaluate_snr(model, X_test, y_test, snr_test):
    model.eval()

    snr_correct = defaultdict(int)
    snr_total = defaultdict(int)

    all_preds, all_targets, all_snrs = [], [], []

    with torch.no_grad():
        for i in range(0, len(X_test), 256):
            x = X_test[i:i+256].to(DEVICE)
            y = y_test[i:i+256].to(DEVICE)
            snr_batch = snr_test[i:i+256]

            out = model(x)
            _, pred = out.max(1)

            pred = pred.cpu().numpy()
            y_np = y.cpu().numpy()

            all_preds.extend(pred)
            all_targets.extend(y_np)
            all_snrs.extend(snr_batch)

            for p, t, s in zip(pred, y_np, snr_batch):
                snr_total[s] += 1
                if p == t:
                    snr_correct[s] += 1

    snrs_sorted = sorted(snr_total.keys())
    accs = [100 * snr_correct[s] / snr_total[s] for s in snrs_sorted]

    return snrs_sorted, accs, all_preds, all_targets, all_snrs

def evaluate_18db(all_preds, all_targets, all_snrs, class_names, exp_name):
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    snrs = np.array(all_snrs)

    mask = snrs == 18
    preds_18 = preds[mask]
    targets_18 = targets[mask]

    acc = (preds_18 == targets_18).mean() * 100
    print(f"{exp_name} → 18 dB Accuracy: {acc:.2f}%")

    cm = confusion_matrix(targets_18, preds_18)

    # Avoid division by zero
    cm = cm.astype(np.float32)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    disp.plot(
        cmap='Blues',
        xticks_rotation=45,
        values_format='.2f'   # show percentages
    )

    plt.title(f"{exp_name} - Normalized Confusion Matrix (18 dB)")

    os.makedirs(f"plots/{exp_name}", exist_ok=True)
    plt.savefig(f"plots/{exp_name}/conf_matrix_18db_normalized.png")
    plt.close()

    return acc

# ================= EXPERIMENT =================
def run_experiment(name, config, train_loader, test_loader, X_test, y_test, snr_test, class_names):

    print(f"\n🚀 Running: {name}")

    model = AMRModel(num_classes=len(class_names), **config).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = test(model, test_loader, criterion)

        print(f"[{name}] Epoch {epoch+1}: Test Acc = {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/{name}.pth")

    # Load best model
    model.load_state_dict(torch.load(f"models/{name}.pth"))

    snrs, accs, preds, targets, snrs_all = evaluate_snr(model, X_test, y_test, snr_test)

    # Save SNR plot
    os.makedirs(f"plots/{name}", exist_ok=True)
    plt.figure()
    plt.plot(snrs, accs, marker='o')
    plt.xlabel("SNR")
    plt.ylabel("Accuracy")
    plt.title(name)
    plt.grid()
    plt.savefig(f"plots/{name}/snr_curve.png")
    plt.close()

    acc_18 = evaluate_18db(preds, targets, snrs_all, class_names, name)

    return best_acc, acc_18

# ================= MAIN =================
def main():

    train_loader, test_loader, class_names, X_test, y_test, snr_test = load_data(
        r'data/RML2016.10a_dict.pkl'
    )

    # 🔬 Ablation configs
    experiments = {
        "baseline": dict(),
        "no_attention": dict(use_attention=False),
        "no_lstm": dict(use_lstm=False),
        "no_residual": dict(use_residual=False),
        "no_depthwise": dict(use_depthwise=False),
        "uni_lstm": dict(bidirectional=False),
    }

    results = []

    for name, config in experiments.items():
        best_acc, acc_18 = run_experiment(
            name, config,
            train_loader, test_loader,
            X_test, y_test, snr_test,
            class_names
        )

        results.append([name, best_acc, acc_18])

    # Save results
    with open("ablation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment", "Best Test Acc", "18dB Acc"])
        writer.writerows(results)

    print("\n📊 Final Results:")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()