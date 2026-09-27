import os
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = "data/risk_dataset.csv"
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"


def prepare_dataset():
    # Load dataset
    df = pd.read_csv(INPUT_PATH)

    print("\n===== DATASET VALIDATION =====")

    # Basic information
    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    # Check required columns
    required_columns = [
        "document_id",
        "text",
        "label",
        "policy_type",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Check duplicate documents
    duplicate_count = df["text"].duplicated().sum()
    print(f"\nDuplicate texts: {duplicate_count}")

    # Show class distribution
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # Show policy distribution
    print("\nPolicy type distribution:")
    print(df["policy_type"].value_counts())

    # Validate labels
    allowed_labels = {"low", "medium", "high"}

    actual_labels = set(df["label"].unique())

    invalid_labels = actual_labels - allowed_labels

    if invalid_labels:
        raise ValueError(
            f"Invalid labels found: {invalid_labels}"
        )

    # Remove accidental empty rows
    df = df.dropna(subset=["text", "label"]).copy()

    # Train/test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    # Create output directory
    os.makedirs("data", exist_ok=True)

    # Save datasets
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("\n===== TRAIN / TEST SPLIT =====")
    print(f"Training records: {len(train_df)}")
    print(f"Testing records: {len(test_df)}")

    print("\nTraining class distribution:")
    print(train_df["label"].value_counts())

    print("\nTesting class distribution:")
    print(test_df["label"].value_counts())

    print("\nDataset preparation completed successfully.")

    print(f"\nTraining dataset: {TRAIN_PATH}")
    print(f"Testing dataset:  {TEST_PATH}")


if __name__ == "__main__":
    prepare_dataset()