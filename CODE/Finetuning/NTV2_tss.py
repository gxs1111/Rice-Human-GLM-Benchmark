import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef, confusion_matrix
from Bio import SeqIO
import random

POSITIVE_TSS_PATH = "<POSITIVE_TSS_FASTA>"
NEGATIVE_TSS_PATH = "<NEGATIVE_TSS_FASTA>"
MODEL_PATH = "<MODEL_PATH>"
VOCAB_PATH = "<VOCAB_PATH>"
OUTPUT_DIR = "<OUTPUT_DIR>"
MAX_LENGTH = 175
BATCH_SIZE = 16
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
LEARNING_RATE = 5e-5
WARMUP_STEPS = 16000
WEIGHT_DECAY = 0.01
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current device: {device}")
print(f"Current GPU: {torch.cuda.current_device()}")

def load_fasta(file_path, label):
    sequences = []
    labels = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq).upper())
        labels.append(label)
    return sequences, labels

positive_tss_sequences, positive_tss_labels = load_fasta(POSITIVE_TSS_PATH, 1)
negative_tss_sequences, negative_tss_labels = load_fasta(NEGATIVE_TSS_PATH, 0)
sequences = positive_tss_sequences + negative_tss_sequences
labels = positive_tss_labels + negative_tss_labels

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

tokenized_lengths = [len(tokenizer.encode(seq, add_special_tokens=True)) for seq in sequences]
max_tokenized_length = max(tokenized_lengths)
print(f"Maximum tokenized length in raw data: {max_tokenized_length}")

def split_dataset(sequences, labels, train_ratio=0.8, val_ratio=0.1):
    data = list(zip(sequences, labels))
    random.shuffle(data)
    sequences, labels = zip(*data)
    total = len(sequences)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return (
        sequences[:train_end], labels[:train_end],
        sequences[train_end:val_end], labels[train_end:val_end],
        sequences[val_end:], labels[val_end:]
    )

train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels = split_dataset(sequences, labels)

train_dataset = Dataset.from_dict({"text": train_sequences, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_sequences, "label": val_labels})
test_dataset = Dataset.from_dict({"text": test_sequences, "label": test_labels})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt"
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2,
    trust_remote_code=True
).to(device)
print(f"Model device: {next(model.parameters()).device}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    recall = recall_score(labels, predictions)
    precision = precision_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    cm = confusion_matrix(labels, predictions).tolist()
    return {
        "accuracy": acc,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "mcc": mcc,
        "confusion_matrix": cm
    }

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3):
        self.patience = patience
        self.best_f1 = -float("inf")
        self.epochs_no_improve = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_f1 = metrics.get("eval_f1", -float("inf"))
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered: no improvement in validation F1 for {self.patience} epochs")
                control.should_training_stop = True
        return control

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    adam_beta1=ADAM_BETA1,
    adam_beta2=ADAM_BETA2,
    adam_epsilon=ADAM_EPSILON,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=100,
    save_total_limit=1,
    disable_tqdm=False,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(patience=EARLY_STOPPING_PATIENCE)]
)

trainer.train()

test_results = trainer.evaluate(test_dataset)
print("Test Metrics:", test_results)

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "test_metrics.txt"), "w") as f:
    f.write("Test Metrics:\n")
    for key, value in test_results.items():
        f.write(f"{key}: {value}\n")
    f.write("\nLabel mapping: {0: Negative TSS, 1: Positive TSS}\n")

trainer.save_model(os.path.join(OUTPUT_DIR, "best_model"))