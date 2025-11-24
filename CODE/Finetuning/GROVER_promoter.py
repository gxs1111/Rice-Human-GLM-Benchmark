import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from Bio import SeqIO
import os

promoter_fasta = "<PROMOTER_FASTA>"
non_promoter_fasta = "<NON_PROMOTER_FASTA>"
pretrained_model_path = "<GROVER_MODEL_PATH>"
output_path = "<OUTPUT_DIR>"
max_length = 256
batch_size = 32
learning_rate = 1e-6
num_epochs = 20
patience = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(output_path):
    os.makedirs(output_path)

log_file = os.path.join(output_path, "training_log.txt")
best_metrics_file = os.path.join(output_path, "best_metrics.txt")

def load_fasta(fasta_path, label):
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        sequences.append(seq)
    labels = [label] * len(sequences)
    return sequences, labels

print("Loading promoter and non-promoter sequences...")
promoter_seqs, promoter_labels = load_fasta(promoter_fasta, 1)
non_promoter_seqs, non_promoter_labels = load_fasta(non_promoter_fasta, 0)

sequences = promoter_seqs + non_promoter_seqs
labels = promoter_labels + non_promoter_labels
data = pd.DataFrame({"sequence": sequences, "class": labels})

print("Checking label distribution...")
promoter_data = data[data["class"] == 1]
non_promoter_data = data[data["class"] == 0]
print(f"Total dataset: Positive = {len(promoter_data)}, Negative = {len(non_promoter_data)}")

train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)
val_data = test_data.sample(frac=0.5, random_state=42)
test_data = test_data.drop(val_data.index)

for split_name, split_data in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
    pos_count = len(split_data[split_data["class"] == 1])
    neg_count = len(split_data[split_data["class"] == 0])
    print(f"{split_name}: Positive = {pos_count}, Negative = {neg_count}")
    if pos_count != neg_count:
        print(f"Warning: {split_name} set is imbalanced (positive:negative ratio != 1:1)")

print(f"Total samples: {len(data)}")
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
    label = data["class"].iloc[i]
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
    print(f"Label: {label}")
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
    def __init__(self, sequences, y, tokenizer, max_length=256):
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

train_dataset = GroverDataset(train_data["sequence"].values, train_data["class"].values, grover_tokenizer, max_length)
val_dataset = GroverDataset(val_data["sequence"].values, val_data["class"].values, grover_tokenizer, max_length)
test_dataset = GroverDataset(test_data["sequence"].values, test_data["class"].values, grover_tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

def val_loop(model, val_loader, device, criterion, epoch, log_file):
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
            pred_labels = (probs[:, 1] > 0.5).astype(int)
            target_labels = target.cpu().numpy()
            preds.extend(pred_labels)
            targets.extend(target_labels)
    val_loss /= len(val_loader)
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    mcc = matthews_corrcoef(targets, preds)

    print(f'\nValidation (Epoch {epoch}): Avg Loss: {val_loss:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'MCC: {mcc:.4f}\n')

    with open(log_file, 'a') as f:
        f.write(f'Epoch {epoch}\n')
        f.write(f'Validation: Avg Loss: {val_loss:.4f}\n')
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(f'Precision: {prec:.4f}\n')
        f.write(f'Recall: {rec:.4f}\n')
        f.write(f'MCC: {mcc:.4f}\n\n')

    return val_loss, acc, f1, prec, rec, mcc

def test_loop(model, test_loader, device, criterion):
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
            pred_labels = (probs[:, 1] > 0.5).astype(int)
            target_labels = target.cpu().numpy()
            preds.extend(pred_labels)
            targets.extend(target_labels)
    test_loss /= len(test_loader)
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    mcc = matthews_corrcoef(targets, preds)

    print(f'Test Set: Avg Loss: {test_loss:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'MCC: {mcc:.4f}\n')

    return test_loss, acc, f1, prec, rec, mcc

print("Starting GROVER fine-tuning...")
grover = BertForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=2).to(device)
optimizer = torch.optim.AdamW(grover.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')
best_val_f1 = 0.0
early_stop_counter = 0
best_f1_epoch = 0

with open(log_file, 'w') as f:
    f.write("Training Log\n\n")

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    train_loss = train_loop(grover, train_loader, epoch, device, optimizer, criterion)
    val_loss, val_acc, val_f1, val_prec, val_rec, val_mcc = val_loop(grover, val_loader, device, criterion, epoch, log_file)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        print("Reset early stopping counter")
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print(f"Early stopping: validation loss did not improve for {patience} epochs")
            if best_val_f1 > 0:
                print(f"Saving best F1 model (epoch {best_f1_epoch}, val F1: {best_val_f1:.4f})")
                test_loss, test_acc, test_f1, test_prec, test_rec, test_mcc = test_loop(grover, test_loader, device, criterion)
                grover_tokenizer.save_pretrained(os.path.join(output_path, "best_f1_model"))
                grover.save_pretrained(os.path.join(output_path, "best_f1_model"))
                with open(best_metrics_file, 'w') as f:
                    f.write(f"Best F1 Model (val epoch {best_f1_epoch}, val F1: {best_val_f1:.4f}):\n")
                    f.write(f"Test Loss: {test_loss:.4f}\n")
                    f.write(f"Test Accuracy: {test_acc:.4f}\n")
                    f.write(f"Test F1 Score: {test_f1:.4f}\n")
                    f.write(f"Test Precision: {test_prec:.4f}\n")
                    f.write(f"Test Recall: {test_rec:.4f}\n")
                    f.write(f"Test MCC: {test_mcc:.4f}\n")
            break

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_f1_epoch = epoch
        test_loss, test_acc, test_f1, test_prec, test_rec, test_mcc = test_loop(grover, test_loader, device, criterion)
        print(f"New best validation F1: {best_val_f1:.4f} (epoch {epoch})")
        grover_tokenizer.save_pretrained(os.path.join(output_path, "best_f1_model"))
        grover.save_pretrained(os.path.join(output_path, "best_f1_model"))
        with open(best_metrics_file, 'w') as f:
            f.write(f"Best F1 Model (val epoch {best_f1_epoch}, val F1: {best_val_f1:.4f}):\n")
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Test F1 Score: {test_f1:.4f}\n")
            f.write(f"Test Precision: {test_prec:.4f}\n")
            f.write(f"Test Recall: {test_rec:.4f}\n")
            f.write(f"Test MCC: {test_mcc:.4f}\n")

if early_stop_counter < patience:
    print(f"Training completed, no early stopping. Saving best F1 model (epoch {best_f1_epoch}, val F1: {best_val_f1:.4f})")
    test_loss, test_acc, test_f1, test_prec, test_rec, test_mcc = test_loop(grover, test_loader, device, criterion)
    grover_tokenizer.save_pretrained(os.path.join(output_path, "best_f1_model"))
    grover.save_pretrained(os.path.join(output_path, "best_f1_model"))
    with open(best_metrics_file, 'w') as f:
        f.write(f"Best F1 Model (val epoch {best_f1_epoch}, val F1: {best_val_f1:.4f}):\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test Precision: {test_prec:.4f}\n")
        f.write(f"Test Recall: {test_rec:.4f}\n")
        f.write(f"Test MCC: {test_mcc:.4f}\n")

print("Training completed!")