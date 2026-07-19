import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from app.predictor import predict


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


def evaluate():

    df = pd.read_csv("dataset/train.csv")

    overall_true = []
    overall_pred = []

    print("=" * 70)
    print("ATTRIBUTE LEVEL EVALUATION")
    print("=" * 70)

    for column in TARGET_COLUMNS:

        y_true = []
        y_pred = []

        for _, row in df.iterrows():

            prediction = predict(row["description"])

            y_true.append(str(row[column]))
            y_pred.append(str(prediction[column]))

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        overall_true.extend(y_true)
        overall_pred.extend(y_pred)

        print(f"\n{column.upper()}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

    overall_accuracy = accuracy_score(
        overall_true,
        overall_pred,
    )

    _, _, overall_f1, _ = precision_recall_fscore_support(
        overall_true,
        overall_pred,
        average="weighted",
        zero_division=0
    )

    print("\n")
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(f"Overall Accuracy : {overall_accuracy:.4f}")
    print(f"Overall F1 Score : {overall_f1:.4f}")


if __name__ == "__main__":
    evaluate()