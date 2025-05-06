import torch
import numpy as np
import pandas as pd
import os # Added for path joining
from config import ENCODER_TYPE, all_dataset_list
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
# ================== START: Added for K-Fold Support ==================
from sklearn.model_selection import StratifiedKFold # Though split happens outside, good to have context
# =================== END: Added for K-Fold Support ===================

class HyoEmoDataSet(Dataset):
    # ================== START MODIFICATION ==================
    # Modified __init__ to accept data directly or load from file
    def __init__(self, dataset, mode=None, data=None, labels=None, tokenizer=None):
        """
        Initializes the Dataset. Can either load data from CSV based on
        dataset and mode, OR use pre-loaded data passed via `data` and `labels`.

        Args:
            dataset (str): Name of the dataset (e.g., 'go_emotion').
            mode (str, optional): 'train', 'valid', or 'test'. Required if data/labels not provided. Defaults to None.
            data (list/pd.Series, optional): Pre-loaded text data. Defaults to None.
            labels (list/pd.Series, optional): Pre-loaded labels corresponding to data. Defaults to None.
            tokenizer (AutoTokenizer, optional): Pre-initialized tokenizer. If None, creates one. Defaults to None.
        """
        super().__init__()
        assert dataset in all_dataset_list
        # Ensure either mode is provided OR data/labels are provided
        if data is not None and labels is not None:
            if not (isinstance(data, (list, pd.Series, np.ndarray)) and isinstance(labels, (list, pd.Series, np.ndarray))):
                 raise ValueError("If providing data/labels, they must be list, pd.Series, or np.ndarray")
            if len(data) != len(labels):
                 raise ValueError("Provided data and labels must have the same length")
            print(f"Initializing HyoEmoDataSet with pre-loaded data (size: {len(data)}).")
            # Use provided data directly
            # Convert to list for consistent __getitem__ indexing if Series/ndarray
            self.text = data.tolist() if isinstance(data, (pd.Series, np.ndarray)) else data
            self.label = labels.tolist() if isinstance(labels, (pd.Series, np.ndarray)) else labels
            self.mode = "preloaded" # Indicate data source
        elif mode is not None:
            assert mode in ['train', 'valid', 'test']
            print(f"Initializing HyoEmoDataSet by loading '{mode}.csv' for dataset '{dataset}'.")
            file_path = os.path.join('data', dataset, f'{mode}.csv')
            try:
                df = pd.read_csv(file_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Error: Data file not found at {file_path}. Ensure './data/{dataset}/{mode}.csv' exists.")
            self.text = df.text.tolist()
            self.label = df.label.tolist() # Assuming labels are already in correct format/type
            self.mode = mode
        else:
            raise ValueError("Must provide either 'mode' (train/valid/test) OR 'data' and 'labels'.")

        # Initialize tokenizer
        if tokenizer:
            self.tkr = tokenizer
            print("Using provided tokenizer.")
        else:
            print(f"Initializing new tokenizer: {ENCODER_TYPE}")
            self.tkr = AutoTokenizer.from_pretrained(ENCODER_TYPE)

    # =================== END MODIFICATION ===================

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # Returns the raw text and label for the collate function to handle
        return self.text[idx], self.label[idx]

    def collate(self, batch):
        """Collator function for DataLoader."""
        texts = [t for t, _ in batch]
        labels = [l for _, l in batch]

        # Tokenize texts
        encode = self.tkr(texts, padding='longest', truncation=True, max_length=200, return_tensors='pt')

        # Convert labels to tensor (assuming labels are numerical or can be directly converted)
        try:
            label_tensor = torch.tensor(labels)
        except Exception as e:
            print(f"Error converting labels to tensor in collate function: {e}")
            print(f"Labels received: {labels}")
            # Handle error appropriately, maybe raise or return None/empty tensor
            raise e # Re-raise error

        return encode, label_tensor


# ================== START: New Function for K-Fold ==================
def load_full_dev_and_test_data(dataset, batch_size, use_bert_specific_dataset=False):
    """
    Loads and combines train/validation data for k-fold cross-validation,
    and loads the separate test data into a DataLoader.

    Args:
        dataset (str): The name of the dataset (e.g., 'go_emotion').
        batch_size (int): Batch size for the test DataLoader.
        use_bert_specific_dataset (bool): Whether to use HyoEmoDataSetForBert (if True)
                                          or HyoEmoDataSet (if False) for the test set.

    Returns:
        tuple: (X_dev, y_dev, test_loader)
            - X_dev (pd.Series): Text data from combined train and validation sets.
            - y_dev (pd.Series): Labels from combined train and validation sets.
            - test_loader (DataLoader): DataLoader for the test set.
    """
    print(f"Loading development (train+valid) and test data for dataset: {dataset}")
    train_path = os.path.join('data', dataset, 'train.csv')
    valid_path = os.path.join('data', dataset, 'valid.csv')
    test_path = os.path.join('data', dataset, 'test.csv')

    try:
        df_train = pd.read_csv(train_path)
        df_valid = pd.read_csv(valid_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading data files: {e}. Ensure train.csv, valid.csv, and test.csv exist in ./data/{dataset}/")

    # Combine train and validation data
    df_dev = pd.concat([df_train, df_valid], ignore_index=True)
    X_dev = df_dev.text # Keep as Series for potential indexing benefits later if needed
    y_dev = df_dev.label # Keep as Series

    print(f"Development data size (train+valid): {len(df_dev)}")
    print(f"Test data size: {len(df_test)}")

    # Prepare test DataLoader
    # Use the appropriate Dataset class based on the flag
    # Initialize tokenizer once to share
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_TYPE)
    print(f"Creating test DataLoader with batch size: {batch_size}")
    if use_bert_specific_dataset:
         # Note: HyoEmoDataSetForBert needs adaptation similar to HyoEmoDataSet
         # if it needs to accept pre-loaded data in the future.
         # For now, assuming it loads 'test' mode correctly.
         print("Using HyoEmoDataSetForBert for test set.")
         test_dataset = HyoEmoDataSetForBert(dataset=dataset, mode='test', encoder_type=ENCODER_TYPE) # Pass appropriate args if needed
         # If HyoEmoDataSetForBert needs a tokenizer, it creates its own currently.
         # Consider passing the shared tokenizer if refactoring HyoEmoDataSetForBert.
    else:
         print("Using HyoEmoDataSet for test set.")
         # Pass the already loaded test data and tokenizer
         test_dataset = HyoEmoDataSet(dataset=dataset, mode=None, # Mode is None because we pass data
                                      data=df_test.text.tolist(),
                                      labels=df_test.label.tolist(),
                                      tokenizer=tokenizer)


    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate)

    # Return combined dev data (as Series) and the test loader
    return X_dev, y_dev, test_loader
# =================== END: New Function for K-Fold ===================


# HyoEmoDataSetForBert remains largely unchanged for now.
# If you plan to use it with k-fold splits in the same way,
# it would need similar modifications to its __init__ to accept
# pre-loaded data (data, labels) instead of only reading from file.
class HyoEmoDataSetForBert(Dataset):
    def __init__(self, dataset, mode, encoder_type='bert-base-uncased', label_included=None, upper_label=None):
        super().__init__()
        assert dataset in all_dataset_list
        assert mode in ['train', 'valid', 'test']
        assert not (label_included and upper_label) # Ensure only one label modification logic is used
        print(f"Initializing HyoEmoDataSetForBert for dataset '{dataset}', mode '{mode}'.")
        file_path = os.path.join('data', dataset, f'{mode}.csv')
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
             raise FileNotFoundError(f"Error: Data file not found at {file_path}.")

        le = LabelEncoder() # LabelEncoder for specific cases

        # Apply label filtering/mapping if specified
        if label_included:
            print(f"Filtering labels to include only: {label_included}")
            df = df.loc[df.label.isin(label_included)].copy() # Use .copy() to avoid SettingWithCopyWarning
            df.reset_index(inplace=True, drop=True) # Reset index after filtering
            # Fit and transform labels based on the filtered subset
            self.label = le.fit_transform(df.label)
            print(f"Labels transformed using LabelEncoder. Mappings: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        elif upper_label:
             print(f"Mapping labels using upper_label dictionary.")
             # Ensure all labels in df.label exist as keys in upper_label for safety
             missing_keys = set(df.label) - set(upper_label.keys())
             if missing_keys:
                  print(f"Warning: Labels {missing_keys} found in data but not in upper_label map. They will result in None.")
             self.label = [upper_label.get(i) for i in df.label]
             # Filter out potential None values if necessary, or handle them later
             # Example: df = df[[l is not None for l in self.label]]
             #          self.label = [l for l in self.label if l is not None]
        else:
            # Use original labels
            self.label = df.label.tolist()

        self.text = df.text.tolist() # Store as list
        print(f"Data loaded. Number of samples: {len(self.text)}")

        # Initialize tokenizer
        print(f"Initializing tokenizer: {encoder_type}")
        self.tkr = AutoTokenizer.from_pretrained(encoder_type)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
         # Check index bounds
         if idx >= len(self.text):
              raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.text)}")
         return self.text[idx], self.label[idx]

    def collate(self, batch):
        """Collator function for DataLoader."""
        texts = [t for t, _ in batch]
        labels = [l for _, l in batch]

        # Tokenize texts
        # Corrected variable name from test_tensor to encode
        encode = self.tkr(texts, padding='longest', truncation=True, max_length=200, return_tensors='pt')

        # Convert labels to tensor
        try:
            # Handle potential non-numeric labels if upper_label or other logic applied
            # Ensure labels are in a numerical format suitable for torch.tensor
            if not all(isinstance(l, (int, float)) for l in labels):
                 # Attempt conversion or raise error if labels are not numeric
                 # This depends on what format labels should be in (e.g., indices)
                 # If LabelEncoder was used, they should be ints. If not, they might be strings.
                 print(f"Warning: Non-numeric labels detected in batch: {labels}. Ensure labels are numerical for tensor conversion.")
                 # Add specific conversion logic here if needed, e.g., mapping strings back to indices
                 # For now, assume they should be numeric and let tensor creation fail if not.
                 pass
            label_tensor = torch.tensor(labels)
        except Exception as e:
             print(f"Error converting labels to tensor in HyoEmoDataSetForBert collate: {e}")
             print(f"Labels received: {labels}")
             raise e

        return encode, label_tensor