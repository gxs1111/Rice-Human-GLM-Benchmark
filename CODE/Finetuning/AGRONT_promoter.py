import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from Bio import SeqIO
import numpy as np
import random
import matplotlib.pyplot as plt

CONFIG = {
    "model_dir": "<MODEL_DIR>",
    "vocab_file": "<VOCAB_FILE>",
    "promoter_fasta": "<PROMOTER_FASTA>",
    "non_promoter_fasta": "<NON_PROMOTER_FASTA>",
    "output_dir": "<OUTPUT_DIR>",
    "logging_dir": "<LOGGING_DIR>",
    "metrics_file": "<METRICS_FILE>",
    "f1_curve_file": "<F1_CURVE_FILE>",
    "maxlength": 85,
    "train_split": 0.8,
    "valid_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42,
    "num_train_epochs": 20,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "logging_steps": 100,
    "save_total_limit": 1,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.0,
    "cuda_devices": 6,
    "metric_for_best_model": "f1",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["logging_dir"], exist_ok=True)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG["cuda_devices"])

print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_dir"], vocab_file=CONFIG["vocab_file"])
print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_dir"],
    num_labels=2,
    ignore_mismatched_sizes=True
)
print(f"Model embedding layer vocabulary size: {model.get_input_embeddings().num_embeddings}")

if tokenizer.vocab_size != model.get_input_embeddings().num_embeddings:
    print(f"Resizing model embedding layer from {model.get_input_embeddings().num_embeddings} to {tokenizer.vocab_size}")
    model.resize_token_embeddings(tokenizer.vocab_size)

def read_fasta(file_path, label):
    sequences = []
    labels = []
    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq).upper()
        if not all(c in 'ATCGN' for c in seq):
            seq = ''.join(c if c in 'ATCGN' else 'N' for c in seq)
        sequences.append(seq)
        labels.append(label)
    return sequences, labels

promoter_seqs, promoter_labels = read_fasta(CONFIG["promoter_fasta"], 1)
non_promoter_seqs, non_promoter_labels = read_fasta(CONFIG["non_promoter_fasta"], 0)

min_samples = min(len(promoter_seqs), len(non_promoter_seqs))
print(f"Balancing dataset: using {min_samples} samples per class")
random.seed(CONFIG["random_seed"])
promoter_indices = random.sample(range(len(promoter_seqs)), min_samples)
non_promoter_indices = random.sample(range(len(non_promoter_seqs)), min_samples)
balanced_promoter_seqs = [promoter_seqs[i] for i in promoter_indices]
balanced_non_promoter_seqs = [non_promoter_seqs[i] for i in non_promoter_indices]
balanced_promoter_labels = [1] * min_samples
balanced_non_promoter_labels = [0] * min_samples

sequences = balanced_promoter_seqs + balanced_non_promoter_seqs
labels = balanced_promoter_labels + balanced_non_promoter_labels
combined = list(zip(sequences, labels))
random.shuffle(combined)
sequences, labels = zip(*combined)

data = {"sequence": sequences, "labels": labels}
dataset = Dataset.from_dict(data)

train_test = dataset.train_test_split(test_size=1 - CONFIG["train_split"], seed=CONFIG["random_seed"])
test_valid = train_test["test"].train_test_split(test_size=CONFIG["test_split"] / (CONFIG["test_split"] + CONFIG["valid_split"]), seed=CONFIG["random_seed"])
train_dataset = train_test["train"]
valid_dataset = test_valid["train"]
test_dataset = test_valid["test"]

def tokenize_function(examples):
    inputs = tokenizer(
        examples["sequence"],
        padding="max_length",
        truncation=True,
        max_length=CONFIG["maxlength"],
        return_tensors="pt"
    )
    max_id = inputs["input_ids"].max().item()
    if max_id >= tokenizer.vocab_size:
        invalid_ids = [id for id in inputs["input_ids"].flatten().tolist() if id >= tokenizer.vocab_size]
        if invalid_ids:
            for invalid_id in set(invalid_ids):
                print(f"Invalid token ID {invalid_id} corresponds to: {tokenizer.convert_ids_to_tokens([invalid_id])[0]}")
    return inputs

train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
valid_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

from collections import Counter
print("Training set label distribution:", Counter(train_dataset["labels"].numpy()))
print("Validation set label distribution:", Counter(valid_dataset["labels"].numpy()))
print("Test set label distribution:", Counter(test_dataset["labels"].numpy()))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions)
    rec = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc
    }

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_train_epochs"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model=CONFIG["metric_for_best_model"],
    greater_is_better=True,
    logging_dir=CONFIG["logging_dir"],
    logging_steps=CONFIG["logging_steps"],
    save_total_limit=CONFIG["save_total_limit"],
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=CONFIG["early_stopping_patience"],
        early_stopping_threshold=CONFIG["early_stopping_threshold"]
    )],
)

trainer.train()

best_model_path = os.path.join configurable["output_dir"], "best_model")
trainer.model.save_pretrained(best_model_path)
tokenizer.save_pretrained(best_model_path)
print(f"Best model saved to: {best_model_path}")

history = trainer.state.log_history
epochs = [log["epoch"] for log in history if "eval_f1" in log]
f1_scores = [log["eval_f1"] for log in history if "eval_f1" in log]
if epochs and f1_scores:
    plt.plot(epochs, f1_scores)
    plt.xlabel("Epoch")
    plt.ylabel("Validation F1 Score")
    plt.title("F1 Score vs. Epoch")
    plt.savefig(CONFIG["f1_curve_file"])
    plt.close()
    print(f"F1 score curve saved to: {CONFIG['f1_curve_file']}")

test_results = trainer.evaluate(test_dataset)

with open(CONFIG["metrics_file"], "w") as f:
    f.write("Best test set metrics:\n")
    for metric, value in test_results.items():
        f.write(f"{metric}: {value:.4f}\n")
print("Best test set metrics saved to:", CONFIG["metrics_file"])