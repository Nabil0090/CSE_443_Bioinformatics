import os
import torch
import numpy as np
import sklearn.metrics
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
from torch.utils.data import Dataset
from genomic_benchmarks.data_check import list_datasets, info
from genomic_benchmarks.loc2seq import download_dataset
from pathlib import Path

class GenomicDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_genomic_benchmark_data(dataset_name, split='train'):
    dataset_path = download_dataset(dataset_name)
    sequences = []
    labels = []
    
    split_path = Path(dataset_path) / split
    class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
    
    for label_idx, class_dir in enumerate(class_dirs):
        for seq_file in class_dir.glob('*.txt'):
            with open(seq_file, 'r') as f:
                sequence = f.read().strip()
                sequences.append(sequence)
                labels.append(label_idx)
    
    return sequences, labels

def calculate_metrics(predictions, labels):
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(valid_labels, valid_predictions),
        "precision": sklearn.metrics.precision_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(valid_labels, valid_predictions, average="macro", zero_division=0),
    }

def preprocess_logits(logits, _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return torch.argmax(logits, dim=-1)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metrics(predictions, labels)

def train_omni_dna_classifier(
    dataset_name="human_nontata_promoters",
    model_name="zehui127/Omni-DNA-116M",
    output_dir="./omni_dna_output",
    seed=42,
    learning_rate=5e-6,
    batch_size=8,
    num_epochs=4,
    max_length=512
):
    print(f"Training Omni-DNA on {dataset_name}")
    
    set_seed(seed)
    
    print("Loading dataset...")
    train_sequences, train_labels = load_genomic_benchmark_data(dataset_name, split='train')
    test_sequences, test_labels = load_genomic_benchmark_data(dataset_name, split='test')
    
    num_classes = len(set(train_labels))
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_sequences)}")
    print(f"Test samples: {len(test_sequences)}")
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = max_length
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        trust_remote_code=True
    )
    
    print("Preparing datasets...")
    train_dataset = GenomicDataset(train_sequences, train_labels, tokenizer, max_length)
    test_dataset = GenomicDataset(test_sequences, test_labels, tokenizer, max_length)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        max_grad_norm=1.0,
        metric_for_best_model="matthews_correlation",
        greater_is_better=True,
        save_total_limit=2,
        load_best_model_at_end=False,
        save_safetensors=False,
        logging_steps=10,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits,
    )
    
    print("Starting training...")
    print("NOTE: Warning about uninitialized weights is expected - they will be trained!")
    trainer.train()
    
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("="*50)
    
    return trainer, model, test_metrics

if __name__ == "__main__":
    print("Available Genomic Benchmark datasets:")
    datasets = list_datasets()
    for i, ds in enumerate(datasets):
        print(f"{i+1}. {ds}")
    
    print("\n" + "="*50)
    print("Starting Omni-DNA Training Pipeline")
    print("="*50 + "\n")
    
    trainer, model, metrics = train_omni_dna_classifier(
        dataset_name="human_nontata_promoters",
        model_name="zehui127/Omni-DNA-116M",
        output_dir="./omni_dna_promoter_classifier",
        seed=42,
        learning_rate=5e-6,
        batch_size=8,
        num_epochs=4,
        max_length=512
    )
    
    print("\nTraining complete! Model saved to ./omni_dna_promoter_classifier")