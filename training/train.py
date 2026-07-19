# import os
# import pandas as pd
# import pickle
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# from datasets import Dataset

# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     TrainingArguments,
#     Trainer
# )

# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support
# )

# df = pd.read_csv("dataset/train.csv")

# df["embellishment"] = df["embellishment"].fillna("None")

# # TARGET = "category"
# # TARGET = "silhouette"
# # TARGET = "fabric"
# # TARGET = "neckline"
# # TARGET = "sleeve"
# # TARGET = "length"
# # TARGET = "embellishment"
# TARGET = "color"

# encoder = LabelEncoder()

# df[TARGET] = encoder.fit_transform(df[TARGET])

# train_df, test_df = train_test_split(
#     df,
#     test_size=0.2,
#     random_state=42,
#     stratify=df[TARGET]
# )

# train_dataset = Dataset.from_pandas(train_df)
# test_dataset = Dataset.from_pandas(test_df)

# tokenizer = AutoTokenizer.from_pretrained(
#     "distilbert-base-uncased"
# )

# os.makedirs("models", exist_ok=True)

# with open(f"models/{TARGET}_encoder.pkl", "wb") as f:
#     pickle.dump(encoder, f)

# def tokenize(batch):
#     return tokenizer(
#         batch["description"],
#         truncation=True,
#         padding="max_length",
#         max_length=64,
#     )

# train_dataset = train_dataset.map(tokenize, batched=True)
# test_dataset = test_dataset.map(tokenize, batched=True)

# train_dataset = train_dataset.rename_column(
#     TARGET,
#     "labels"
# )

# test_dataset = test_dataset.rename_column(
#     TARGET,
#     "labels"
# )

# keep_columns = [
#     "input_ids",
#     "attention_mask",
#     "labels"
# ]

# train_dataset.set_format(
#     type="torch",
#     columns=keep_columns
# )

# test_dataset.set_format(
#     type="torch",
#     columns=keep_columns
# )

# model = AutoModelForSequenceClassification.from_pretrained(
#     "distilbert-base-uncased",
#     num_labels=len(encoder.classes_)
# )

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred

#     predictions = np.argmax(logits, axis=-1)

#     precision, recall, f1, _ = precision_recall_fscore_support(
#         labels,
#         predictions,
#         average="weighted"
#     )

#     accuracy = accuracy_score(labels, predictions)

#     return {
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#     }
    
# training_args = TrainingArguments(
#     output_dir=f"models/{TARGET}_model",

#     eval_strategy="epoch",

#     save_strategy="epoch",

#     learning_rate=2e-5,

#     per_device_train_batch_size=8,

#     per_device_eval_batch_size=8,

#     num_train_epochs=3,

#     weight_decay=0.01,

#     load_best_model_at_end=True,

#     # logging_dir="logs",

#     logging_steps=10,

#     report_to="none"
# )

# trainer = Trainer(
#     model=model,

#     args=training_args,

#     train_dataset=train_dataset,

#     eval_dataset=test_dataset,

#     compute_metrics=compute_metrics,
# )

# trainer.train()

# results = trainer.evaluate()

# print("\nEvaluation Results")

# for key, value in results.items():
#     print(f"{key}: {value}")

# trainer.save_model(f"models/{TARGET}_model")

# tokenizer.save_pretrained(f"models/{TARGET}_model")




import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"

DATASET_PATH = "dataset/train.csv"

MODEL_DIR = "models"

MAX_LENGTH = 64

RANDOM_STATE = 42

EPOCHS = 3

LEARNING_RATE = 2e-5

TRAIN_BATCH_SIZE = 8

EVAL_BATCH_SIZE = 8

# Attributes to train
TARGET_COLUMNS = [
    "category",
    "silhouette",
    "fabric",
    "neckline",
    "sleeve",
    "length",
    "embellishment",
    "color",
]

# Create models directory
os.makedirs(MODEL_DIR, exist_ok=True)

# Load tokenizer only once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

# Fill missing values
df["embellishment"] = df["embellishment"].fillna("None")

print(f"Dataset Shape : {df.shape}")

print("\nMissing Values")

print(df.isnull().sum())

print("=" * 60)


# ---------------------------------------------------
# Metrics
# ---------------------------------------------------

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for HuggingFace Trainer.
    """

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------
# Tokenizer Function
# ---------------------------------------------------

def tokenize(batch):
    """
    Tokenize descriptions.
    """

    return tokenizer(
        batch["description"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


# ---------------------------------------------------
# Save JSON
# ---------------------------------------------------

def save_json(data, filepath):
    """
    Save dictionary as JSON.
    """

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
        # ---------------------------------------------------
# Train One Attribute
# ---------------------------------------------------

def train_attribute(target):

    print("\n" + "=" * 60)
    print(f"Training Attribute : {target.upper()}")
    print("=" * 60)

    # -----------------------------------------------
    # Copy Dataset
    # -----------------------------------------------

    temp_df = df.copy()

    # -----------------------------------------------
    # Label Encoding
    # -----------------------------------------------

    encoder = LabelEncoder()

    temp_df[target] = encoder.fit_transform(
        temp_df[target]
    )

    # Save Encoder
    encoder_path = os.path.join(
        MODEL_DIR,
        f"{target}_encoder.pkl"
    )

    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)

    print(f"Encoder Saved : {encoder_path}")

    # -----------------------------------------------
    # Train Test Split
    # -----------------------------------------------

    train_df, test_df = train_test_split(
        temp_df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=temp_df[target]
    )

    print(f"Training Samples : {len(train_df)}")
    print(f"Testing Samples  : {len(test_df)}")

    # -----------------------------------------------
    # Convert to HuggingFace Dataset
    # -----------------------------------------------

    train_dataset = Dataset.from_pandas(
        train_df
    )

    test_dataset = Dataset.from_pandas(
        test_df
    )

    # -----------------------------------------------
    # Tokenization
    # -----------------------------------------------

    train_dataset = train_dataset.map(
        tokenize,
        batched=True
    )

    test_dataset = test_dataset.map(
        tokenize,
        batched=True
    )

    # -----------------------------------------------
    # Rename Label Column
    # -----------------------------------------------

    train_dataset = train_dataset.rename_column(
        target,
        "labels"
    )

    test_dataset = test_dataset.rename_column(
        target,
        "labels"
    )

    # -----------------------------------------------
    # Torch Format
    # -----------------------------------------------

    keep_columns = [
        "input_ids",
        "attention_mask",
        "labels",
    ]

    train_dataset.set_format(
        type="torch",
        columns=keep_columns
    )

    test_dataset.set_format(
        type="torch",
        columns=keep_columns
    )

    print("Dataset Ready For Training")
        # -----------------------------------------------
    # Load Model
    # -----------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(encoder.classes_)
    )

    # -----------------------------------------------
    # Training Arguments
    # -----------------------------------------------

    output_path = os.path.join(
        MODEL_DIR,
        f"{target}_model"
    )

    training_args = TrainingArguments(
        output_dir=output_path,

        eval_strategy="epoch",

        save_strategy="epoch",

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,

        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        num_train_epochs=EPOCHS,

        weight_decay=0.01,

        load_best_model_at_end=True,

        logging_steps=10,

        report_to="none",
    )

    # -----------------------------------------------
    # Trainer
    # -----------------------------------------------

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=test_dataset,

        compute_metrics=compute_metrics,
    )

    # -----------------------------------------------
    # Train Model
    # -----------------------------------------------

    print(f"\nTraining {target} model...\n")

    trainer.train()

    # -----------------------------------------------
    # Evaluate
    # -----------------------------------------------

    print("\nEvaluating Model...")

    results = trainer.evaluate()

    print("\nEvaluation Results")

    for key, value in results.items():
        print(f"{key} : {value}")

    # -----------------------------------------------
    # Save Model
    # -----------------------------------------------

    trainer.save_model(output_path)

    tokenizer.save_pretrained(output_path)

    # -----------------------------------------------
    # Save Metrics
    # -----------------------------------------------

    metrics = {
        "attribute": target,
        "accuracy": float(results.get("eval_accuracy", 0)),
        "precision": float(results.get("eval_precision", 0)),
        "recall": float(results.get("eval_recall", 0)),
        "f1": float(results.get("eval_f1", 0))
    }

    metrics_path = os.path.join(
        MODEL_DIR,
        f"{target}_metrics.json"
    )

    save_json(metrics, metrics_path)

    print(f"\nModel Saved : {output_path}")

    print(f"Metrics Saved : {metrics_path}")

    print("=" * 60)
    # ---------------------------------------------------
# Main Function
# ---------------------------------------------------

def main():

    print("\n")
    print("=" * 70)
    print("PRODUCT ATTRIBUTE EXTRACTION TRAINING")
    print("=" * 70)

    all_metrics = []

    for target in TARGET_COLUMNS:

        train_attribute(target)

        metrics_file = os.path.join(
            MODEL_DIR,
            f"{target}_metrics.json"
        )

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        all_metrics.append(metrics)

    print("\n")
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    for metric in all_metrics:

        print(
            f"{metric['attribute']:<18}"
            f" Accuracy: {metric['accuracy']:.4f}"
            f" | F1: {metric['f1']:.4f}"
        )

    overall_accuracy = np.mean(
        [m["accuracy"] for m in all_metrics]
    )

    overall_f1 = np.mean(
        [m["f1"] for m in all_metrics]
    )

    summary = {
        "overall_accuracy": float(overall_accuracy),
        "overall_f1": float(overall_f1),
        "attributes": all_metrics
    }

    save_json(
        summary,
        os.path.join(
            MODEL_DIR,
            "training_summary.json"
        )
    )

    print("\n")
    print("=" * 70)
    print(f"Overall Accuracy : {overall_accuracy:.4f}")
    print(f"Overall F1 Score : {overall_f1:.4f}")
    print("=" * 70)

    print("\nTraining Completed Successfully!")
    print(f"All models saved inside '{MODEL_DIR}/'")


# ---------------------------------------------------
# Entry Point
# ---------------------------------------------------

if __name__ == "__main__":
    main()