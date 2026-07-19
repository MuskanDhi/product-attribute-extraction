# Product Attribute Extraction using DistilBERT

## Overview

This project implements an AI/NLP pipeline that extracts structured product attributes from unstructured fashion product descriptions using Hugging Face's DistilBERT model.

The system takes a product description as input and predicts multiple product attributes such as category, fabric, silhouette, neckline, sleeve type, dress length, embellishment, and color.

The project also provides a FastAPI endpoint for real-time attribute extraction.

---

## Features

- Generate a labeled dataset of fashion product descriptions.
- Train DistilBERT models for attribute classification.
- Extract structured attributes from product descriptions.
- REST API using FastAPI.
- Evaluation using Accuracy, Precision, Recall, and F1 Score.
- Swagger API documentation.

---

## Technologies Used

- Python 3.11+
- Hugging Face Transformers
- DistilBERT
- PyTorch
- Scikit-learn
- Pandas
- FastAPI
- Uvicorn

---

## Project Structure

```
product-attribute-extraction/
│
├── app/
│   ├── main.py
│   └── predictor.py
│
├── dataset/
│   └── train.csv
│
├── models/
│   ├── category_model/
│   ├── silhouette_model/
│   ├── fabric_model/
│   ├── neckline_model/
│   ├── sleeve_model/
│   ├── length_model/
│   ├── embellishment_model/
│   ├── color_model/
│   ├── *.pkl
│   └── *.json
│
├── training/
│   ├── generate_dataset.py
│   ├── preprocess.py
│   └── train.py
│
├── evaluation.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The dataset consists of **100 labeled product descriptions** generated for training.

Each record contains the following fields:

- Description
- Category
- Silhouette
- Fabric
- Neckline
- Sleeve
- Length
- Embellishment
- Color

Example:

| Description                                           | Category         | Fabric  | Color |
| ----------------------------------------------------- | ---------------- | ------- | ----- |
| Floor length chiffon bridesmaid dress with V neckline | Bridesmaid Dress | Chiffon | Sage  |

---

## Model

Model Used:

**distilbert-base-uncased**

Each attribute is trained as a separate multi-class classification model.

Attributes:

- Category
- Silhouette
- Fabric
- Neckline
- Sleeve
- Length
- Embellishment
- Color

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project

```bash
cd product-attribute-extraction
```

Create virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Generate Dataset

```bash
python training/generate_dataset.py
```

---

## Preprocess Dataset

```bash
python training/preprocess.py
```

---

## Train Models

```bash
python training/train.py
```

After training, the trained models and label encoders will be saved inside the `models/` directory.

---

## Run Evaluation

```bash
python evaluation.py
```

---

## Run API

```bash
uvicorn app.main:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## API Endpoint

### POST /extract

### Request

```json
{
  "text": "Lace mermaid wedding dress with long sleeves and scalloped hem"
}
```

### Response

```json
{
  "description": "Lace mermaid wedding dress with long sleeves and scalloped hem",
  "attributes": {
    "category": "Wedding Dress",
    "silhouette": "Empire",
    "fabric": "Satin",
    "neckline": "One Shoulder",
    "sleeve": "Long Sleeve",
    "length": "Midi",
    "embellishment": "Beaded",
    "color": "Gold"
  }
}
```

---

## Evaluation Results

| Attribute     | Accuracy | F1 Score |
| ------------- | -------- | -------- |
| Category      | 96.00%   | 95.96%   |
| Silhouette    | 31.00%   | 17.02%   |
| Fabric        | 16.00%   | 5.80%    |
| Neckline      | 18.00%   | 9.49%    |
| Sleeve        | 27.00%   | 15.88%   |
| Length        | 53.00%   | 38.49%   |
| Embellishment | 24.00%   | 14.36%   |
| Color         | 27.00%   | 16.37%   |

### Overall Performance

- Overall Accuracy: **36.50%**
- Overall F1 Score: **26.67%**

---

## Common Failure Cases

The model performs well for the **Category** attribute but has lower performance on some other attributes due to the limited size of the dataset.

Common failure cases include:

- Confusion between similar silhouettes (e.g., Mermaid vs. Trumpet).
- Incorrect prediction of colors when multiple colors are mentioned.
- Difficulty identifying embellishments if not explicitly stated.
- Limited generalization because the dataset contains only 100 synthetic samples.

Increasing the dataset size and diversity would likely improve the model's performance.

---

## Future Improvements

- Train on a larger dataset (500–1000+ samples).
- Use a single multi-task learning model to predict all attributes simultaneously.
- Add confidence scores for predictions.
- Support batch prediction.
- Deploy the API to a cloud platform such as Render, Railway, or AWS.

---

## Author

**Muskan Dhiman**

B.Tech CSE

Product Attribute Extraction Assignment
