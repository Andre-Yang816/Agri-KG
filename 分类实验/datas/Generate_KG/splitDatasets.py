import json
import random
from sklearn.model_selection import train_test_split

# Example: Load your data (this is your original JSON-like structure)
filename = 'GeneraKG.json'
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Function to split the dataset into train, test, and validation sets
def split_data(data, train_size=0.6, val_size=0.2, test_size=0.2, random_seed=42):
    # Ensure the sum of train, val, and test sizes equals 1
    assert train_size + val_size + test_size == 1, "Sizes must sum to 1"

    # Shuffle the data
    random.seed(random_seed)
    random.shuffle(data)

    # Split into train+val and test first
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=random_seed)

    # Split the train+val data into train and validation sets
    train_data, val_data = train_test_split(train_val_data, test_size=val_size / (train_size + val_size),
                                            random_state=random_seed)

    return train_data, val_data, test_data

# Split the dataset
train_data, val_data, test_data = split_data(data)

# Optionally save the split data into JSON files for later use
with open("train_data.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open("val_data.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)

with open("test_data.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

# Print out the size of each set for verification
print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")
