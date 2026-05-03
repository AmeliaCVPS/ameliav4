"""
train_bert_triage.py - Fine-tuning BERT para triagem Manchester
================================================================

Treina classificacao direta:
  texto livre -> BERT -> verde/amarelo/laranja/vermelho

Exemplo:
  python train_bert_triage.py --data data/generated/triagem_massiva_texto.jsonl --epochs 3

O modelo padrao e neuralmind/bert-base-portuguese-cased. Se houver um
BioBERTpt/clinico local ou no HuggingFace, passe em --base-model.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABEL2ID = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
PT2EN = {"verde": "green", "amarelo": "yellow", "laranja": "orange", "vermelho": "red"}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def normalize_label(value: str) -> str:
    value = str(value).strip().lower()
    return PT2EN.get(value, value)


def read_dataset(path: str, text_col: str, label_col: str, sample: int | None) -> pd.DataFrame:
    if path.endswith(".jsonl") or path.endswith(".jsonl.gz"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)

    if text_col not in df.columns:
        raise ValueError(f"Coluna de texto '{text_col}' nao encontrada. Colunas: {list(df.columns)}")
    if label_col not in df.columns:
        # compatibilidade com o gerador
        label_col = "classificacao" if "classificacao" in df.columns else "risk_color"

    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    df["label"] = df["label"].map(normalize_label)
    df = df[df["label"].isin(LABEL2ID)]
    if sample:
        df = df.sample(min(sample, len(df)), random_state=42)
    return df.reset_index(drop=True)


class TriageTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    df = read_dataset(args.data, args.text_col, args.label_col, args.sample)
    print(f"Dataset: {len(df):,} exemplos")
    print(df["label"].value_counts())

    y = df["label"].map(LABEL2ID).astype(int).to_numpy()
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = TriageTextDataset(
        train_df["text"].astype(str).tolist(),
        train_df["label"].map(LABEL2ID).astype(int).tolist(),
        tokenizer,
        args.max_length,
    )
    test_ds = TriageTextDataset(
        test_df["text"].astype(str).tolist(),
        test_df["label"].map(LABEL2ID).astype(int).tolist(),
        tokenizer,
        args.max_length,
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2, 3]),
        y=train_df["label"].map(LABEL2ID).astype(int).to_numpy(),
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=args.fp16,
        report_to="none",
        save_total_limit=2,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights_t,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    preds = trainer.predict(test_ds)
    y_true = test_df["label"].map(LABEL2ID).astype(int).to_numpy()
    y_pred = np.argmax(preds.predictions, axis=-1)
    print(classification_report(
        y_true,
        y_pred,
        target_names=[ID2LABEL[i] for i in range(4)],
    ))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, ensure_ascii=False, indent=2)
    print(f"Modelo salvo em: {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV/JSONL gerado")
    parser.add_argument("--output-dir", default="models/bert_triage")
    parser.add_argument("--base-model", default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--text-col", default="texto")
    parser.add_argument("--label-col", default="risk_color")
    parser.add_argument("--sample", type=int, default=None, help="amostra opcional para teste rapido")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
