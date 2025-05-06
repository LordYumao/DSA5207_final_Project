# convert_to_hypemo.py
import pandas as pd
import os
import csv # <--- Added import for quoting control
# from config import LABEL2ID # Assuming LABEL2ID might be defined elsewhere if not here

# Assuming LABEL2ID is defined somewhere in the code (or defined here as provided)
LABEL2ID = {
    "joy": 0,
    "sadness": 1,
    "fear": 2,
    "anger": 3,
    "disgust": 4,
    "anticipation": 5,
    "optimism": 6,
    "pessimism": 7,
    "love": 8,
    'surprise': 9,
    'trust': 10
    # Add other labels if necessary
}

def convert_to_hypemo(input_tsv, output_tsv):
    """
    Converts an input TSV with string labels to HypEmo's intermediate TSV format.
    input_tsv:   path/to/…_single_label.tsv (Expected columns: id, text, label)
    output_tsv:  path/to/…_hypemo.tsv (Output format: processed_text<TAB>label_id)
    """
    try:
        # Try reading with expected columns, handle potential missing 'id'
        try:
            df = pd.read_csv(input_tsv, sep="\t", encoding="utf8", usecols=["text", "label"])
        except ValueError:
            print(f"Warning: Columns 'text', 'label' not found or other issue in {input_tsv}. Attempting to read first two columns.")
            df = pd.read_csv(input_tsv, sep="\t", encoding="utf8", header=None, names=["text", "label"], usecols=[0, 1])

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_tsv}")
        return
    except Exception as e:
        print(f"Error reading {input_tsv}: {e}")
        return

    processed_count = 0
    skipped_count = 0
    with open(output_tsv, "w", encoding="utf8") as fout:
        for _, row in df.iterrows():
            # Handle potential NaN/missing values
            label_str = str(row.get("label", "")).strip().lower()
            text_str = str(row.get("text", ""))

            if label_str in ("", "none", "nan"):
                skipped_count += 1
                continue # Skip rows with empty, "none", or NaN labels

            if label_str not in LABEL2ID:
                print(f"Warning: Label '{label_str}' not found in LABEL2ID mapping. Skipping row.")
                skipped_count += 1
                continue # Skip rows with unknown labels

            idx = LABEL2ID[label_str]
            # Basic text cleaning: replace tabs and newlines with spaces
            text = text_str.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            fout.write(f"{text}\t{idx}\n")
            processed_count += 1

    print(f"Converted {input_tsv} → {output_tsv}. Processed: {processed_count}, Skipped: {skipped_count}")


def hypemo_tsv_to_csv(src_tsv, dest_csv):
    """
    Convert a *_hypemo.tsv file (text<TAB>label_id) to the target CSV format
    with columns: text, aug_text, label.
    'aug_text' will be a copy of 'text' as a placeholder.
    """
    try:
        # Read the intermediate TSV (text<TAB>label_id)
        df = pd.read_csv(src_tsv, sep="\t", header=None, names=["text", "label"], quoting=csv.QUOTE_NONE)

        # --- Modification Start ---
        # Create the 'aug_text' column as a placeholder (copy of 'text')
        df['aug_text'] = df['text']

        # Reorder columns to match the target format: text, aug_text, label
        df = df[['text', 'aug_text', 'label']]
        # --- Modification End ---

        # Save to the final CSV format
        # index=False: Don't write row numbers
        # quoting=csv.QUOTE_MINIMAL: Add quotes only when necessary (e.g., if text contains commas)
        df.to_csv(dest_csv, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Converted {src_tsv} → {dest_csv} (with columns: text, aug_text, label)")

    except FileNotFoundError:
        print(f"Error: Input file not found: {src_tsv}")
    except pd.errors.EmptyDataError:
         print(f"Error: Input file {src_tsv} is empty.")
    except Exception as e:
        print(f"Error converting {src_tsv} to {dest_csv}: {e}")


if __name__ == "__main__":
    # Define base path (adjust if your script is elsewhere relative to data)
    base_data_path = "data/semeval2018_ec_single/"
    output_data_path = "data/se2018_single/"
    os.makedirs(base_data_path, exist_ok=True) # Ensure directory exists
    os.makedirs(output_data_path, exist_ok=True)

    # Define input filenames (assuming these exist)
    train_input_tsv = os.path.join(base_data_path, "2018-E-c-En-train_single_label.tsv")
    dev_input_tsv = os.path.join(base_data_path, "2018-E-c-En-dev_single_label.tsv") # Assuming dev is used for test set

    # Define intermediate filenames
    train_hypemo_tsv = os.path.join(base_data_path, "train_hypemo.tsv") # Simplified name
    dev_hypemo_tsv = os.path.join(base_data_path, "dev_hypemo.tsv") # Simplified name

    # Define final output CSV filenames
    train_final_csv = os.path.join(output_data_path, "train.csv")
    test_final_csv = os.path.join(output_data_path, "valid.csv") # Using dev data as test data

    # --- Step 1: Convert original TSV to intermediate HypEmo TSV ---
    print("--- Starting Step 1: Convert to HypEmo Intermediate TSV ---")
    convert_to_hypemo(train_input_tsv, train_hypemo_tsv)
    convert_to_hypemo(dev_input_tsv, dev_hypemo_tsv) # Convert dev set as well

    # --- Step 2: Convert intermediate HypEmo TSV to final 3-column CSV ---
    print("\n--- Starting Step 2: Convert HypEmo TSV to Final 3-Column CSV ---")
    hypemo_tsv_to_csv(train_hypemo_tsv, train_final_csv)
    hypemo_tsv_to_csv(dev_hypemo_tsv, test_final_csv) # Create test.csv from dev data

    print("\nConversion process finished.")