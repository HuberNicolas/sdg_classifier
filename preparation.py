import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datasets import Dataset, DatasetDict
from skmultilearn.model_selection import iterative_train_test_split

df = pd.read_csv("./data/sdg_data_abstracts_cleaned.csv")
sdg_columns = [f"SDG{i:02d}" for i in range(1,18,1)]
print(df.shape) # (18153, 19)


if df[sdg_columns].isnull().values.any():
    print("NaN values found in SDG columns. Filling with 0.")
    df[sdg_columns] = df[sdg_columns].fillna(0)
df[sdg_columns] = df[sdg_columns].apply(pd.to_numeric, errors='coerce')

df = df.drop('Unnamed: 0', axis=1)
df.head()
#df.to_csv("./data/sdg_data_abstracts_cleaned.csv")


#os.mkdir("./output")

def plot_target_distribution(df, sdg_columns):
    target_counts = df[sdg_columns].sum()
    plt.figure(figsize=(12, 6))
    target_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of SDG Labels', fontsize=16)
    plt.xlabel('SDG Categories', fontsize=14)
    plt.ylabel('Number of Occurrences', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("./output/distribution.pdf")
    #plt.show()

plot_target_distribution(df=df, sdg_columns=sdg_columns)



# Features (abstract) and Labels (SDGs)
X = df[["abstract"]].values
y = df.iloc[:, 1:].values
print(y.dtype)  # Check data type
print(np.unique(y))  # Check unique values in `y`

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
train_df = pd.concat([train_df, pd.DataFrame(y_train, columns=df.columns[1:])], axis=1)

val_df = pd.DataFrame(X_val, columns=["abstract"])
val_df = pd.concat([val_df, pd.DataFrame(y_val, columns=df.columns[1:])], axis=1)

test_df = pd.DataFrame(X_test, columns=["abstract"])
test_df = pd.concat([test_df, pd.DataFrame(y_test, columns=df.columns[1:])], axis=1)

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

# Save or inspect
print(dataset_dict)

# Save to disk
dataset_dict.save_to_disk("./data/preprocessed/sdg_dataset_splits_multilabel")