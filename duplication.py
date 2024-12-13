import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from skmultilearn.model_selection import iterative_train_test_split

# Load dataset
df = pd.read_csv("./data/sdg_data_abstracts_cleaned.csv")
sdg_columns = [f"SDG{i:02d}" for i in range(1, 18, 1)]

# Check for NaN and fill with 0
if df[sdg_columns].isnull().values.any():
    print("NaN values found in SDG columns. Filling with 0.")
    df[sdg_columns] = df[sdg_columns].fillna(0)
df[sdg_columns] = df[sdg_columns].apply(pd.to_numeric, errors='coerce')

# Drop unnecessary columns
df = df.drop('Unnamed: 0', axis=1, errors='ignore')

# Function to plot target distribution
def plot_target_distribution(df, sdg_columns, title, output_path):
    target_counts = df[sdg_columns].sum()
    plt.figure(figsize=(12, 6))
    target_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(title, fontsize=16)
    plt.xlabel('SDG Categories', fontsize=14)
    plt.ylabel('Number of Occurrences', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# Plot original target distribution
plot_target_distribution(df, sdg_columns, "Original Distribution of SDG Labels", "./output/pre_duplicated_distribution.pdf")

# Duplicate rows for each SDG label
def duplicate_rows(df, sdg_columns):
    duplicated_data = []
    for _, row in df.iterrows():
        for sdg in sdg_columns:
            if row[sdg] == 1:
                new_row = row.copy()
                new_row[sdg_columns] = 0  # Set all SDGs to 0
                new_row[sdg] = 1          # Set the specific SDG to 1
                duplicated_data.append(new_row)
    return pd.DataFrame(duplicated_data)

duplicated_df = duplicate_rows(df, sdg_columns)

# Plot duplicated target distribution
plot_target_distribution(duplicated_df, sdg_columns, "Duplicated Distribution of SDG Labels", "./output/duplicated_distribution.pdf")

# Print summary
print("Original dataset size:", df.shape)
print("Duplicated dataset size:", duplicated_df.shape)

# Features (abstract) and Labels (SDGs)
X = duplicated_df[["abstract"]].values
y = duplicated_df[sdg_columns].values

# First Split: Train and Temp (for further splitting into val/test)
X_train, y_train, X_temp, y_temp = iterative_train_test_split(
    X, y, test_size=0.4
)

# Second Split: Validation and Test
X_val, y_val, X_test, y_test = iterative_train_test_split(
    X_temp, y_temp, test_size=0.5
)

# Convert splits to DataFrames
train_df = pd.DataFrame(X_train, columns=["abstract"])
train_df = pd.concat([train_df, pd.DataFrame(y_train, columns=sdg_columns)], axis=1)

val_df = pd.DataFrame(X_val, columns=["abstract"])
val_df = pd.concat([val_df, pd.DataFrame(y_val, columns=sdg_columns)], axis=1)

test_df = pd.DataFrame(X_test, columns=["abstract"])
test_df = pd.concat([test_df, pd.DataFrame(y_test, columns=sdg_columns)], axis=1)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Save to disk
dataset_dict.save_to_disk("./data/preprocessed/sdg_dataset_splits_multilabel_duplicated")

print("Dataset successfully saved!")
