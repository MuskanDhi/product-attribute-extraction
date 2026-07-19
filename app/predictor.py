import os
import pickle
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
MODEL_DIR = "models"

MODEL_NAME = "distilbert-base-uncased"

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
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
models = {}

encoders = {}

for target in TARGET_COLUMNS:

    model_path = os.path.join(
        MODEL_DIR,
        f"{target}_model"
    )

    encoder_path = os.path.join(
        MODEL_DIR,
        f"{target}_encoder.pkl"
    )

    models[target] = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    models[target].eval()

    with open(encoder_path, "rb") as f:
        encoders[target] = pickle.load(f)

print("All Models Loaded Successfully")
def predict(description):

    predictions = {}

    inputs = tokenizer(
        description,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64,
    )

    with torch.no_grad():

        for target in TARGET_COLUMNS:

            outputs = models[target](**inputs)

            prediction = torch.argmax(
                outputs.logits,
                dim=1
            ).item()

            value = encoders[target].inverse_transform(
                [prediction]
            )[0]

            predictions[target] = value

    return predictions
if __name__ == "__main__":

    sample = (
        "Floor length chiffon bridesmaid dress "
        "with pleated bodice and V neckline "
        "available in sage and dusty blue"
    )

    result = predict(sample)

    print("\nPrediction")

    for key, value in result.items():
        print(f"{key:15} : {value}")