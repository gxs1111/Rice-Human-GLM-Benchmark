import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

DATA_FILE = "<DATA_CSV>"
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

class SpeciesDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def sequence_to_onehot(seq, max_len=MAX_SEQ_LEN):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, max_len), dtype=np.float32)
    for i, base in enumerate(seq[:max_len]):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
        else:
            one_hot[:, i] = 0.25
    return one_hot

def prepare_data(data_file):
    df = pd.read_csv(data_file)
    
    unique_species = sorted(df['species'].unique())
    num_classes = len(unique_species)
    print(f"Unique species: {unique_species}")
    print(f"Number of classes: {num_classes}")

    le = LabelEncoder()
    labels_encoded = le.fit_transform(df['species'])

    sequences = df['sequence'].str.upper().tolist()

    sequences_onehot = [sequence_to_onehot(seq) for seq in sequences]
    sequences_onehot = np.array(sequences_onehot)

    sequences_tensor = torch.tensor(sequences_onehot, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)

    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences_tensor, labels_tensor, test_size=0.2, random_state=RANDOM_SEED, stratify=labels_tensor
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )

    train_dataset = SpeciesDataset(X_train, y_train)
    val_dataset = SpeciesDataset(X_val, y_val)
    test_dataset = SpeciesDataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset, num_classes, le

class SpeciesCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpeciesCNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)

        self.fc_input_dim = 256 * (MAX_SEQ_LEN // 8)
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

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
        val_true = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        val_acc = np.mean(np.array(val_preds) == np.array(val_true))
        val_precision = precision_score(val_true, val_preds, average='macro')
        val_recall = recall_score(val_true, val_preds, average='macro')
        val_f1 = f1_score(val_true, val_preds, average='macro')
        val_mcc = matthews_corrcoef(val_true, val_preds)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
            print(f"New best model saved | Val Macro F1: {val_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping: no improvement in validation macro F1 for {patience} epochs")
                break

        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f} | Macro P: {val_precision:.4f} | R: {val_recall:.4f} | "
              f"F1: {val_f1:.4f} | MCC: {val_mcc:.4f}")

        with open(os.path.join(OUTPUT_DIR, 'training_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | "
                    f"Val Acc: {val_acc:.4f} | Macro F1: {val_f1:.4f} | MCC: {val_mcc:.4f}\n")

def evaluate_model(model, test_loader, device, le):
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
    precision = precision_score(all_true, all_preds, average='macro')
    recall = recall_score(all_true, all_preds, average='macro')
    f1 = f1_score(all_true, all_preds, average='macro')
    mcc = matthews_corrcoef(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds)

    print("\n=== Final Test Results (Macro Average) ===")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1 Score:  {f1:.4f}")
    print(f"MCC:             {mcc:.4f}")
    print(f"Species order: {list(le.classes_)}")
    print("Confusion Matrix:")
    print(cm)

    result_str = (f"Test Accuracy: {acc:.4f}\n"
                  f"Macro Precision: {precision:.4f}\n"
                  f"Macro Recall: {recall:.4f}\n"
                  f"Macro F1: {f1:.4f}\n"
                  f"MCC: {mcc:.4f}\n"
                  f"Species order: {list(le.classes_)}\n"
                  f"Confusion Matrix:\n{cm}\n")
    
    with open(os.path.join(OUTPUT_DIR, 'test_results.txt'), 'w') as f:
        f.write(result_str)

def main():
    train_dataset, val_dataset, test_dataset, num_classes, label_encoder = prepare_data(DATA_FILE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SpeciesCNN(num_classes).to(device)

    print(f"Training on {device} | Classes: {num_classes} | Train samples: {len(train_dataset)}")

    train_model(model, train_loader, val_loader, device)

    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path}")

    evaluate_model(model, test_loader, device, label_encoder)

if __name__ == '__main__':
    main()