import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def stratified_split(data, split_col, train_size, test_size, seed):

    # Obtain split column if it exists
    stratify_col = data[split_col] if split_col else None

    # Split the data into train and val-test datasets
    temp_size = 1.0 - train_size
    train_df, temp_df = train_test_split(
        data,
        train_size=train_size,
        stratify=stratify_col,
        random_state=seed
    )

    # Get split column from val-testing data
    stratify_col = temp_df[split_col] if split_col else None

    # Split the val-test data into validation and testing datasets
    val_size = test_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_size,
        stratify=stratify_col,
        random_state=seed
    )

    return train_df, val_df, test_df

def main():
    parser = argparse.ArgumentParser(description="Stratified data splitter")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the input csv file")
    parser.add_argument("--save-path", type=str, required=True, help="Directory to save the split csvs")
    parser.add_argument("--train-size", type=float, default=0.8, help="Proportion of training data")
    parser.add_argument("--test-size", type=float, default=0.1, help="Proportion of test data")
    parser.add_argument("--split-col", type=str, default=None, help="Column name to stratify on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data_path)

    # Split
    train_df, val_df, test_df = stratified_split(
        data,
        split_col=args.split_col,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed
    )

    # Make sure save path exists
    os.makedirs(args.save_path, exist_ok=True)

    print("Splits complete:")
    print(f"Train: {len(train_df)} rows")
    print(f"Val:   {len(val_df)} rows")
    print(f"Test:  {len(test_df)} rows")

    # Save files
    train_df.to_csv(os.path.join(args.save_path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.save_path, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.save_path, "test.csv"), index=False)


if __name__ == "__main__":
    main()
