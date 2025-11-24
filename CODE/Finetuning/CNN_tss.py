import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import random
import os

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Configuration
TSS_POSITIVE_FILE = "<TSS_POSITIVE_FASTA>"
TSS_NEGATIVE_FILE = "<TSS_NEGATIVE_FASTA>"
OUTPUT_DIR = "<OUTPUT_DIR>"
RANDOM_SEED = 42
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64
DROPOUT_RATE = 0.5
MAX_SEQ_LEN = 500
PATIENCE = 3

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Check GPU
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check GPU environment.")
else:
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset class
class TSSDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Read FASTA file
def read_fasta(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    sequences = []
    with open(file_path, 'r') as f:
        seq = ''
        for line in f:
            if line.startswith('>'):
                if seq:
                    sequences.append(seq.upper())
                    seq = ''
            else:
                seq += line.strip()
        if seq:
            sequences.append(seq.upper())
    return sequences

# One-hot encoding
def sequence_to_onehot(seq, max_len=MAX_SEQ_LEN):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, max_len), dtype=np.float32)
    for i, base in enumerate(seq[:max_len]):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
        else:
            one_hot[:, i] = 0.25  # Equal probability for unknown bases
    return one_hot

# Data preparation
def prepare_data(positive_file, negative_file):
    positive_seqs = read_fasta(positive_file)
    negative_seqs = read_fasta(negative_file)

    positive_labels = [1] * len(positive_seqs)
    negative_labels = [0] * len(negative_seqs)

    sequences = positive_seqs + negative_seqs
    labels = positive_labels + negative_labels

    sequences_onehot = [sequence_to_onehot(seq) for seq in sequences]
    sequences_onehot = np.array(sequences_onehot)

    sequences_tensor = torch.tensor(sequences_onehot, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Train/val/test split: 80%/10%/10%
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences_tensor, labels_tensor, test_size=0.2, random_state=RANDOM_SEED, stratify=labels_tensor
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )

    train_dataset = TSSDataset(X_train, y_train)
    val_dataset = TSSDataset(X_val, y_val)
    test_dataset = TSSDataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset

# CNN Model
class TSSCNN(nn.Module):
    def __init__(self):
        super(TSSCNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)

        self.fc_input_dim = 256 * (MAX_SEQ_LEN // 8)
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, val_loader, device, epochs=EPOCHS, patience=PATIENCE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        val_acc = np.mean(np.array(val_preds) == np.array(val_true))
        val_precision = precision_score(val_true, val_preds, pos_label=1)
        val_recall = recall_score(val_true, val_preds, pos_label=1)
        val_f1 = f1_score(val_true, val_preds, pos_label=1)
        val_mcc = matthews_corrcoef(val_true, val_preds)

        # Early stopping & best model saving
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
            print(f"New best model saved | Val F1: {val_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping: no improvement in validation F1 for {patience} epochs")
                break

        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f} | Precision: {val_precision:.4f} | "
              f"Recall: {val_recall:.4f} | F1: {val_f1:.4f} | MCC: {val_mcc:.4f}")

        with open(os.path.join(OUTPUT_DIR, 'training_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | "
                    f"Val Acc: {val_acc:.4f} | F1: {val_f1:.4f} | MCC: {val_mcc:.4f}\n")

# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_true))
    precision = precision_score(all_true, all_preds, pos_label=1)
    recall = recall_score(all_true, all_preds, pos_label=1)
    f1 = f1_score(all_true, all_preds, pos_label=1)
    mcc = matthews_corrcoef(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds)

    print("\n=== Final Test Results ===")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print(f"MCC:          {mcc:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    result_str = (f"Test Accuracy: {acc:.4f}\n"
                  f"Precision: {precision:.4f}\n"
                  f"Recall: {recall:.4f}\n"
                  f"F1 Score: {f1:.4f}\n"
                  f"MCC: {mcc:.4f}\n"
                  f"Confusion Matrix (0=Non-TSS, 1=TSS):\n{cm}\n")

    with open(os.path.join(OUTPUT_DIR, 'test_results.txt'), 'w') as f:
        f.write(result_str)

    return acc, precision, recall, f1, mcc

# Main
def main():
    train_dataset, val_dataset, test_dataset = prepare_data(TSS_POSITIVE_FILE, TSS_NEGATIVE_FILE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TSSCNN().to(device)

    print(f"Training on {device} | Train samples: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_model(model, train_loader, val_loader, device)

    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path}")

    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()