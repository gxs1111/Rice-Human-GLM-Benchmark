import os
import csv
import json
import logging
import random
import numpy as np
import torch
import transformers
import sklearn
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from peft import LoraConfig, get_peft_model
from transformers import EarlyStoppingCallback
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

MODEL_PATH = "<MODEL_PATH>"
DATA_PATH = "<DATA_CSV>"
OUTPUT_DIR = "<OUTPUT_DIR>"
EPOCH = 20
MAX_LENGTH = 128
LEARNING_RATE = 3e-5
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001
RANDOM_SEED = 42

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=MODEL_PATH)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="query,value")

@dataclass
class DataArguments:
    data_path: str = field(default=DATA_PATH)
    kmer: int = field(default=-1)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="DNABERT2_species")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=MAX_LENGTH)
    gradient_accumulation_steps: int = field(default=GRAD_ACCUM_STEPS)
    per_device_train_batch_size: int = field(default=TRAIN_BATCH_SIZE)
    per_device_eval_batch_size: int = field(default=EVAL_BATCH_SIZE)
    num_train_epochs: int = field(default=EPOCH)
    fp16: bool = field(default=True)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=200)
    eval_steps: int = field(default=200)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=LEARNING_RATE)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_f1")
    greater_is_better: bool = field(default=True)
    output_dir: str = field(default=OUTPUT_DIR)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    seed: int = field(default=RANDOM_SEED)
    overwrite_output_dir: bool = field(default=True)
    log_level: str = field(default="info")

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def generate_kmer_str(sequence: str, k: int) -> str:
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            json.dump(kmer, f)
    return kmer

def split_dataset(data_path: str, output_dir: str, train_ratio=0.8, val_ratio=0.1):
    with open(data_path, "r") as f:
        data = list(csv.reader(f))[1:]
    sequences = [d[0] for d in data]
    species_labels = [d[1] for d in data]

    species_to_id = {
        "Bos_taurus": 0,
        "Canis_lupus_familiaris": 1,
        "Homo_sapiens": 2,
        "Mus_musculus": 3,
        "Sus_scrofa": 4
    }
    numeric_labels = [species_to_id[label] for label in species_labels]

    sequences = np.array(sequences)
    numeric_labels = np.array(numeric_labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(sss.split(sequences, numeric_labels))

    train_val_sequences = sequences[train_val_idx]
    train_val_labels = numeric_labels[train_val_idx]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio/(train_ratio + val_ratio), random_state=RANDOM_SEED)
    train_idx, val_idx = next(sss.split(train_val_sequences, train_val_labels))

    train_data = [[sequences[train_val_idx[i]], species_labels[train_val_idx[i]]] for i in train_idx]
    val_data = [[sequences[train_val_idx[i]], species_labels[train_val_idx[i]]] for i in val_idx]
    test_data = [[sequences[i], species_labels[i]] for i in test_idx]

    os.makedirs(output_dir, exist_ok=True)
    for split, split_data in [("train.csv", train_data), ("dev.csv", val_data), ("test.csv", test_data)]:
        with open(os.path.join(output_dir, split), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence", "species"])
            writer.writerows(split_data)

    return os.path.join(output_dir, "train.csv"), os.path.join(output_dir, "dev.csv"), os.path.join(output_dir, "test.csv")

class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, kmer: int = -1):
        super(SupervisedDataset, self).__init__()
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        texts = [d[0] for d in data]
        species_to_id = {
            "Bos_taurus": 0,
            "Canis_lupus_familiaris": 1,
            "Homo_sapiens": 2,
            "Mus_musculus": 3,
            "Sus_scrofa": 4
        }
        labels = [species_to_id[d[1]] for d in data]

        if kmer != -1:
            texts = load_or_generate_kmer(data_path, texts, kmer)

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = 5

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], attention_mask=self.attention_mask[i], labels=torch.tensor(self.labels[i]))

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    cm = confusion_matrix(valid_labels, valid_predictions)
    species_names = ["Bos_taurus", "Canis_lupus_familiaris", "Homo_sapiens", "Mus_musculus", "Sus_scrofa"]
    cm_dict = {"confusion_matrix": cm.tolist(), "labels": species_names}

    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(valid_labels, valid_predictions),
        "precision": sklearn.metrics.precision_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "confusion_matrix": cm_dict
    }

def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return torch.argmax(logits, dim=-1)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)

def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    set_seed(RANDOM_SEED)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.data_path = DATA_PATH
    training_args.output_dir = OUTPUT_DIR

    train_path, val_path, test_path = split_dataset(data_args.data_path, training_args.output_dir)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=train_path, kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=val_path, kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=test_path, kmer=data_args.kmer)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=5,
        trust_remote_code=True,
    )

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    callbacks = [EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=EARLY_STOPPING_THRESHOLD
    )]

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )

    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    train()