import json
import os
import torch
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from standalone_hyenadna import HyenaDNAModel, CharacterTokenizer
from transformers import PreTrainedModel, PretrainedConfig
import re
from tqdm import tqdm

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

WEIGHTS_PATH = "<WEIGHTS_CKPT>"
DATA_PATH = "<DATA_CSV>"
OUTPUT_DIR = "<OUTPUT_DIR>"
MAX_LENGTH = 1024
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EARLY_STOP_PATIENCE = 3
DEVICE = 'cuda:6' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 3
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

def inject_substring(orig_str):
    modified_string = re.sub(r"\.mixer", ".mixer.layer", orig_str)
    modified_string = re.sub(r"\.mlp", ".mlp.layer", modified_string)
    return modified_string

def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            key_loaded = 'model.' + key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except:
                raise Exception('Key mismatch in the state dicts!')
    return scratch_dict

class HyenaDNAConfig(PretrainedConfig):
    model_type = "hyenadna"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class RegionDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(seq, truncation=True, max_length=self.max_length, padding='max_length')
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        return input_ids, label

class HyenaDNAPreTrainedModel(PreTrainedModel):
    config_class = HyenaDNAConfig
    base_model_prefix = "hyenadna"

    def __init__(self, config, use_head=True, n_classes=NUM_CLASSES):
        super().__init__(config)
        self.model = HyenaDNAModel(**config.__dict__, use_head=use_head, n_classes=n_classes)

    def forward(self, input_ids, labels=None):
        outputs = self.model(input_ids)
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(outputs, labels)
            return loss, outputs
        return outputs

    @classmethod
    def from_pretrained(cls, weights_path, config, device='cpu', use_head=True, n_classes=NUM_CLASSES):
        config_dict = json.loads(config) if isinstance(config, str) else config
        config = HyenaDNAConfig(**config_dict)
        model = cls(config, use_head=use_head, n_classes=n_classes)
        loaded_ckpt = torch.load(weights_path, map_location=torch.device(device))
        checkpointing = config_dict.get("checkpoint_mixer", False)
        state_dict = load_weights(model.model.state_dict(), loaded_ckpt['state_dict'], checkpointing=checkpointing)
        model.model.load_state_dict(state_dict)
        print("Loaded pretrained weights successfully!")
        return model

set_seed(RANDOM_SEED)

data = pd.read_csv(DATA_PATH)
sequences = data['sequence'].tolist()
labels = data['label'].tolist()

label_mapping = {"exon": 0, "intron": 1, "intergenic": 2}
labels = np.array([label_mapping[label] for label in labels])
print("Label Mapping:", label_mapping)

train_seq, temp_seq, train_labels, temp_labels = train_test_split(
    sequences, labels, test_size=0.2, stratify=labels, random_state=RANDOM_SEED
)
val_seq, test_seq, val_labels, test_labels = train_test_split(
    temp_seq, temp_labels, test_size=0.5, stratify=temp_labels, random_state=RANDOM_SEED
)

tokenizer = CharacterTokenizer(
    characters=['A', 'C', 'G', 'T', 'N'],
    model_max_length=MAX_LENGTH + 2,
    add_special_tokens=False,
    padding_side='left'
)

train_dataset = RegionDataset(train_seq, train_labels, tokenizer, MAX_LENGTH)
val_dataset = RegionDataset(val_seq, val_labels, tokenizer, MAX_LENGTH)
test_dataset = RegionDataset(test_seq, test_labels, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

config_path = os.path.join(os.path.dirname(WEIGHTS_PATH), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

model = HyenaDNAPreTrainedModel.from_pretrained(
    WEIGHTS_PATH,
    config=config,
    device=DEVICE,
    use_head=True,
    n_classes=NUM_CLASSES
)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

best_f1 = 0.0
patience_counter = 0
best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pt')

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        input_ids, labels = batch
        input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss, _ = model(input_ids, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_preds, val_true = [], []
    with torch.inference_mode():
        for batch in val_loader:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            _, logits = model(input_ids, labels)
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    val_f1 = f1_score(val_true, val_preds, average='macro')
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val F1 (macro): {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with F1: {best_f1:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(torch.load(best_model_path))
model.eval()
test_preds, test_true = [], []
with torch.inference_mode():
    for batch in test_loader:
        input_ids, labels = batch
        input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
        _, logits = model(input_ids, labels)
        preds = torch.argmax(logits, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

acc = accuracy_score(test_true, test_preds)
f1 = f1_score(test_true, test_preds, average='macro')
precision = precision_score(test_true, test_preds, average='macro')
recall = recall_score(test_true, test_preds, average='macro')
mcc = matthews_corrcoef(test_true, test_preds)
cm = confusion_matrix(test_true, test_preds)

metrics = {
    'accuracy': float(acc),
    'f1_score': float(f1),
    'precision': float(precision),
    'recall': float(recall),
    'mcc': float(mcc),
    'confusion_matrix': cm.tolist(),
    'label_mapping': label_mapping
}
with open(os.path.join(OUTPUT_DIR, 'test_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

print("Test Metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(f"Best model saved to: {best_model_path}")
print(f"Test metrics saved to: {os.path.join(OUTPUT_DIR, 'test_metrics.json')}")