import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

# Load dataset
df = pd.read_csv("dataset/train.csv")

print("Dataset Loaded")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nMissing Values Before")
print(df.isnull().sum())

# Fill missing values
df["embellishment"] = df["embellishment"].fillna("None")

print("\nMissing Values After")
print(df.isnull().sum())

# Encode labels
label_encoders = {}

label_columns = [
    "category",
    "silhouette",
    "fabric",
    "neckline",
    "sleeve",
    "length",
    "embellishment",
    "color"
]

for column in label_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder

print("\nLabels Encoded Successfully")

# Train/Test Split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(train_df))
print("Testing Samples:", len(test_df))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

print("\nTokenizer Loaded")

# Tokenize one sample
sample = train_df.iloc[0]["description"]

tokens = tokenizer(
    sample,
    truncation=True,
    padding="max_length",
    max_length=64
)

print("\nSample Description:")
print(sample)

print("\nFirst 20 Token IDs:")
print(tokens["input_ids"][:20])