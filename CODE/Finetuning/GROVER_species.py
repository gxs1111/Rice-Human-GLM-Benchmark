import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

data_file = "<DATA_CSV>"
pretrained_model_path = "<GROVER_MODEL_PATH>"
output_path = "<OUTPUT_DIR>"
max_length = 512
batch_size = 32
learning_rate = 1e-6
num_epochs = 20
patience = 3
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

if not os.path.exists(output_path):
    os.makedirs(output_path)

log_file = os.path.join(output_path, "training_log.txt")
best_metrics_file = os.path.join(output_path, "best_metrics.txt")
confusion_matrix_file = os.path.join(output_path, "confusion_matrix.png")

print("Loading species_human.csv data...")
data = pd.read_csv(data_file)
print(f"Total samples: {len(data)}")

print("Encoding species labels...")
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['species'])
num_classes = len(label_encoder.classes_)
expected_labels = ['Bos_taurus', 'Canis_lupus_familiaris', 'Homo_sapiens', 'Mus_musculus', 'Sus_scrofa']
if list(label_encoder.classes_) != expected_labels:
    print(f"Warning: Actual classes {list(label_encoder.classes_)} do not match expected classes {expected_labels}!")
print(f"Species classes: {label_encoder.classes_}")
print(f"Number of classes: {num_classes}")

print("Checking label distribution...")
label_counts = data['label'].value_counts()
for label, count in label_counts.items():
    print(f"Species {label_encoder.inverse_transform([label])[0]}: {count} samples")

train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)
val_data = test_data.sample(frac=0.5, random_state=42)
test_data = test_data.drop(val_data.index)

for split_name, split_data in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
    pos_counts = split_data['label'].value_counts()
    print(f"\n{split_name} label distribution:")
    for label, count in pos_counts.items():
        print(f"Species {label_encoder.inverse_transform([label])[0]}: {count} samples")
    if len(pos_counts) != num_classes:
        print(f"Warning: {split_name} set does not contain all classes!")

print(f"\nTotal samples: {len(data)}")
print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")

print("Loading GROVER tokenizer...")
try:
    grover_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        pretrained_model_path,
        do_lower_case=False,
        model_max_length=max_length
    )
except Exception as e:
    print(f"Failed to load tokenizer: {e}")
    exit()

print("\nInspecting tokenization of first 3 samples...")
for i in range(min(3, len(data))):
    seq = data["sequence"].iloc[i]
    label = data["label"].iloc[i]
    species = label_encoder.inverse_transform([label])[0]
    tokenized = grover_tokenizer(
        seq,
        add_special_tokens=True,
        padding="max_length",
        return_tensors="pt",
        max_length=max_length,
        truncation=True
    )
    input_ids = tokenized["input_ids"].squeeze(0).tolist()
    attention_mask = tokenized["attention_mask"].squeeze(0).tolist()
    decoded_seq = grover_tokenizer.decode(input_ids, skip_special_tokens=False)

    print(f"\nSample {i+1}:")
    print(f"Original sequence: {seq[:50]}..." if len(seq) > 50 else f"Original sequence: {seq}")
    print(f"Tokenized input_ids: {input_ids[:10]}..." if len(input_ids) > 10 else f"Tokenized input_ids: {input_ids}")
    print(f"Attention mask: {attention_mask[:10]}..." if len(attention_mask) > 10 else f"Attention mask: {attention_mask}")
    print(f"Species: {species} (encoded: {label})")
    print(f"Decoded sequence: {decoded_seq[:50]}..." if len(decoded_seq) > 50 else f"Decoded sequence: {decoded_seq}")

print("\nComputing token length distribution...")
token_lengths = []
for seq in data["sequence"]:
    tokenized = grover_tokenizer(
        seq,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=False
    )
    token_length = tokenized["input_ids"].shape[1]
    token_lengths.append(token_length)

max_token_length = max(token_lengths)
over_length_count = sum(1 for length in token_lengths if length > max_length)
print(f"Maximum token length: {max_token_length}")
print(f"Samples exceeding {max_length} tokens: {over_length_count} ({100 * over_length_count / len(data):.2f}%)")

class GroverDataset(Dataset):
    def __init__(self, sequences, y, tokenizer, max_length=512):
        print("Loading GROVER dataset...")
        self.tokenizer = tokenizer
        self.sequences = sequences
        self.max_length = max_length
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokenizer_res = self.tokenizer(
            seq,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        )
        ids = tokenizer_res["input_ids"].squeeze(0)
        attention_masks = tokenizer_res["attention_mask"].squeeze(0)
        return ids, self.y[idx], attention_masks, idx

train_dataset = GroverDataset(train_data["sequence"].values, train_data["label"].values, grover_tokenizer, max_length)
val_dataset = GroverDataset(val_data["sequence"].values, val_data["label"].values, grover_tokenizer, max_length)
test_dataset = GroverDataset(test_data["sequence"].values, test_data["label"].values, grover_tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def plot_confusion_matrix(targets, preds, classes, output_path):
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

def train_loop(model, train_loader, epoch, device, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (data, target, attention_masks, _) in enumerate(train_loader):
        data, target, attention_masks = data.to(device), target.to(device), attention_masks.to(device)
        optimizer.zero_grad()
        logits = model(data, attention_mask=attention_masks).logits
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]    Loss: {loss.item():.6f}')
    return total_loss / len(train_loader)

def val_loop(model, val_loader, device, criterion, epoch, log_file, classes):
    model.eval()
    val_loss = 0
    preds = []
    targets = []
    with torch.no_grad():
        for data, target, attention_masks, _ in val_loader:
            data, target, attention_masks = data.to(device), target.to(device), attention_masks.to(device)
            logits = model(data, attention_mask=attention_masks).logits
            val_loss += criterion(logits, target).item()
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            pred_labels = np.argmax(probs, axis=1)
            target_labels = target.cpu().numpy()
            preds.extend(pred_labels)
            targets.extend(target_labels)
    val_loss /= len(val_loader)
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')
    prec = precision_score(targets, preds, average='weighted')
    rec = recall_score(targets, preds, average='weighted')
    mcc = matthews_corrcoef(targets, preds)

    print(f'\nValidation (Epoch {epoch}): Avg Loss: {val_loss:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score (weighted): {f1:.4f}')
    print(f'Precision (weighted): {prec:.4f}')
    print(f'Recall (weighted): {rec:.4f}')
    print(f'MCC: {mcc:.4f}\n')

    with open(log_file, 'a') as f:
        f.write(f'Epoch {epoch}\n')
        f.write(f'Validation: Avg Loss: {val_loss:.4f}\n')
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'F1 Score (weighted): {f1:.4f}\n')
        f.write(f'Precision (weighted): {prec:.4f}\n')
        f.write(f'Recall (weighted): {rec:.4f}\n')
        f.write(f'MCC: {mcc:.4f}\n\n')

    plot_confusion_matrix(targets, preds, classes, os.path.join(output_path, f"val_confusion_matrix_epoch_{epoch}.png"))

    return val_loss, acc, f1, prec, rec, mcc

def test_loop(model, test_loader, device, criterion, classes):
    model.eval()
    test_loss = 0
    preds = []
    targets = []
    with torch.no_grad():
        for data, target, attention_masks, _ in test_loader:
            data, target, attention_masks = data.to(device), target.to(device), attention_masks.to(device)
            logits = model(data, attention_mask=attention_masks).logits
            test_loss += criterion(logits, target).item()
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            pred_labels = np.argmax(probs, axis=1)
            target_labels = target.cpu().numpy()
            preds.extend(pred_labels)
            targets.extend(target_labels)
    test_loss /= len(test_loader)
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')
    prec = precision_score(targets, preds, average='weighted')
    rec = recall_score(targets, preds, average='weighted')
    mcc = matthews_corrcoef(targets, preds)

    print(f'\nTest Set: Avg Loss: {test_loss:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score (weighted): {f1:.4f}')
    print(f'Precision (weighted): {prec:.4f}')
    print(f'Recall (weighted): {rec:.4f}')
    print(f'MCC: {mcc:.4f}\n')

    plot_confusion_matrix(targets, preds, classes, confusion_matrix_file)

    return test_loss, acc, f1, prec, rec, mcc

print("Starting GROVER fine-tuning...")
grover = BertForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=num_classes).to(device)
optimizer = torch.optim.AdamW(grover.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')
best_val_f1 = 0.0
best_metrics = {}
best_test_metrics = {}
early_stop_counter = 0
best_f1_epoch = 0

with open(log_file, 'w') as f:
    f.write("Training Log\n\n")

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    train_loss = train_loop(grover, train_loader, epoch, device, optimizer, criterion)
    val_loss, acc, f1, prec, rec, mcc = val_loop(grover, val_loader, device, criterion, epoch, log_file, label_encoder.classes_)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        print("Reset early stopping counter")
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{patience}")

    if f1 > best_val_f1:
        best_val_f1 = f1
        best_f1_epoch = epoch
        best_metrics = {
            'val_loss': val_loss,
            'acc': acc,
            'f1': f1,
            'prec': prec,
            'rec': rec,
            'mcc': mcc
        }
        print(f"New best validation F1 score: {best_val_f1:.4f} (Epoch {epoch})")
        grover_tokenizer.save_pretrained(os.path.join(output_path, "best_f1_model"))
        grover.save_pretrained(os.path.join(output_path, "best_f1_model"))
        test_loss, test_acc, test_f1, test_prec, test_rec, test_mcc = test_loop(grover, test_loader, device, criterion, label_encoder.classes_)
        best_test_metrics = {
            'test_loss': test_loss,
            'acc': test_acc,
            'f1': test_f1,
            'prec': test_prec,
            'rec': test_rec,
            'mcc': test_mcc
        }
        with open(best_metrics_file, 'w') as f:
            f.write(f"Best Model (Epoch {best_f1_epoch}) - Test Metrics:\n")
            f.write(f"Test Loss: {best_test_metrics['test_loss']:.4f}\n")
            f.write(f"Accuracy: {best_test_metrics['acc']:.4f}\n")
            f.write(f"F1 Score (weighted): {best_test_metrics['f1']:.4f}\n")
            f.write(f"Precision (weighted): {best_test_metrics['prec']:.4f}\n")
            f.write(f"Recall (weighted): {best_test_metrics['rec']:.4f}\n")
            f.write(f"MCC: {best_test_metrics['mcc']:.4f}\n")
            f.write(f"\nValidation Metrics (reference):\n")
            f.write(f"Validation Loss: {best_metrics['val_loss']:.4f}\n")
            f.write(f"Accuracy: {best_metrics['acc']:.4f}\n")
            f.write(f"F1 Score (weighted): {best_metrics['f1']:.4f}\n")
            f.write(f"Precision (weighted): {best_metrics['prec']:.4f}\n")
            f.write(f"Recall (weighted): {best_metrics['rec']:.4f}\n")
            f.write(f"MCC: {best_metrics['mcc']:.4f}\n")

    if early_stop_counter >= patience:
        print(f"Early stopping: validation loss did not improve for {patience} epochs")
        print(f"Saving best F1 model (Epoch {best_f1_epoch}, F1: {best_val_f1:.4f})")
        break

if early_stop_counter < patience:
    print(f"Training completed, no early stopping. Saving best F1 model (Epoch {best_f1_epoch}, F1: {best_val_f1:.4f})")
    if not best_test_metrics:
        test_loss, test_acc, test_f1, test_prec, test_rec, test_mcc = test_loop(grover, test_loader, device, criterion, label_encoder.classes_)
        best_test_metrics = {
            'test_loss': test_loss,
            'acc': test_acc,
            'f1': test_f1,
            'prec': test_prec,
            'rec': test_rec,
            'mcc': test_mcc
        }
        with open(best_metrics_file, 'w') as f:
            f.write(f"Best Model (Epoch {best_f1_epoch}) - Test Metrics:\n")
            f.write(f"Test Loss: {best_test_metrics['test_loss']:.4f}\n")
            f.write(f"Accuracy: {best_test_metrics['acc']:.4f}\n")
            f.write(f"F1 Score (weighted): {best_test_metrics['f1']:.4f}\n")
            f.write(f"Precision (weighted): {best_test_metrics['prec']:.4f}\n")
            f.write(f"Recall (weighted): {best_test_metrics['rec']:.4f}\n")
            f.write(f"MCC: {best_test_metrics['mcc']:.4f}\n")

print("Training completed!")