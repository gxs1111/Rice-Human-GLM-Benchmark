import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

PROMOTER_FILE = "<PROMOTER_FASTA>"
NONPROMOTER_FILE = "<NONPROMOTER_FASTA>"
OUTPUT_DIR = "<OUTPUT_DIR>"
RANDOM_SEED = 42
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64
DROPOUT_RATE = 0.5
MAX_SEQ_LEN = 500
PATIENCE = 3

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check GPU environment.")
else:
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

class PromoterDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

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

def sequence_to_onehot(seq, max_len=MAX_SEQ_LEN):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, max_len), dtype=np.float32)
    for i, base in enumerate(seq[:max_len]):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
        else:
            one_hot[:, i] = 0.25
    return one_hot

def prepare_data(promoter_file, nonpromoter_file):
    promoter_seqs = read_fasta(promoter_file)
    nonpromoter_seqs = read_fasta(nonpromoter_file)

    promoter_labels = [1] * len(promoter_seqs)
    nonpromoter_labels = [0] * len(nonpromoter_seqs)

    sequences = promoter_seqs + nonpromoter_seqs
    labels = promoter_labels + nonpromoter_labels

    sequences_onehot = [sequence_to_onehot(seq) for seq in sequences]
    sequences_onehot = np.array(sequences_onehot)

    sequences_tensor = torch.tensor(sequences_onehot, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences_tensor, labels_tensor, test_size=0.2, random_state=RANDOM_SEED, stratify=labels_tensor
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )

    train_dataset = PromoterDataset(X_train, y_train)
    val_dataset = PromoterDataset(X_val, y_val)
    test_dataset = PromoterDataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset

class PromoterCNN(nn.Module):
    def __init__(self):
        super(PromoterCNN, self).__init__()
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

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        val_precision = precision_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_mcc = matthews_corrcoef(val_labels, val_preds)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
            print(f"New best model saved with Val F1: {val_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered: no improvement in validation F1 for {patience} epochs")
                break

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f} | Precision: {val_precision:.4f} | "
              f"Recall: {val_recall:.4f} | F1: {val_f1:.4f} | MCC: {val_mcc:.4f}")

        with open(os.path.join(OUTPUT_DIR, 'training_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | "
                    f"Val Acc: {val_acc:.4f} | F1: {val_f1:.4f} | MCC: {val_mcc:.4f}\n")

def evaluate_model(model, test_loader, device):
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_mcc = matthews_corrcoef(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)

    print("\n=== Final Test Results ===")
    print(f"Accuracy:   {test_acc:.4f}")
    print(f"Precision:  {test_precision:.4f}")
    print(f"Recall:     {test_recall:.4f}")
    print(f"F1 Score:   {test_f1:.4f}")
    print(f"MCC:        {test_mcc:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    result_str = (f"Test Accuracy: {test_acc:.4f}\n"
                  f"Precision: {test_precision:.4f}\n"
                  f"Recall: {test_recall:.4f}\n"
                  f"F1: {test_f1:.4f}\n"
                  f"MCC: {test_mcc:.4f}\n"
                  f"Confusion Matrix:\n{cm}\n")
    with open(os.path.join(OUTPUT_DIR, 'test_results.txt'), 'w') as f:
        f.write(result_str)

    return test_acc, test_precision, test_recall, test_f1, test_mcc

def main():
    train_dataset, val_dataset, test_dataset = prepare_data(PROMOTER_FILE, NONPROMOTER_FILE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PromoterCNN().to(device)

    print(f"Training on {device} with {len(train_dataset)} training samples")

    train_model(model, train_loader, val_loader, device)

    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path}")

    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()