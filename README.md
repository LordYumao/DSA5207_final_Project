# HypEmo on SE2018 Dataset

This project adapts the HypEmO model from the ACL 2023 paper "[Label-Aware Hyperbolic Embeddings for Fine-grained Emotion Classification](https://aclanthology.org/2023.acl-long.613/)" for use with the "[SemEval-2018 Task 1: Affect in Tweets](https://aclanthology.org/S18-1001/)" (single-label subset). It also includes scripts for training and evaluating standard baseline models (BERT, RoBERTa, ELECTRA) on the same dataset for comparison.

This code is tested under Python 3.10.

### Installation

First, install the necessary packages. It is recommended to use a `requirements.txt` file.

```bash
pip install -r requirements.txt
```
Ensure you have at least torch, transformers, datasets, scikit-learn, pandas, numpy, and accelerate installed.

### Data Preparation
Place the preprocessed SemEval-2018 single-label dataset files (train.csv, valid.csv) into the data/se2018_single/ directory.
The label mapping for this dataset (11 labels) is defined in config_se18.py, which is used by the baseline training script.
Running the HypEmO model might require pre-trained hyperbolic label embeddings. If you need to generate these for the se2018_single dataset, refer to the "Note" section regarding train_label_embedding.py. Pre-computed embeddings (e.g., se2018_single.bin) are typically stored in the label_tree/ directory.

### Running Experiments
## 1. Running Baseline Models (BERT, RoBERTa, ELECTRA, etc.):

Use the train_evaluate_baseline.py script.

Configuration: Before running, you can modify the MODEL_NAME (e.g., "roberta-base", "bert-large-uncased") and SEED variables directly within the script, or pass them as command-line arguments if the script is set up to use argparse (e.g., using --model_name and --seed).
Execution (example using args):
```Bash
python train_evaluate_baseline.py --model_name roberta-base --seed 111 --dataset_name se2018_single
```
Multiple Runs: To replicate the paper's methodology for reporting mean and standard deviation, you need to run the script 5 times for each baseline model, using a different SEED value (e.g., 111, 222, 333, 444, 555) for each run while keeping other hyperparameters constant.

## 2. Running the HypEmO Model:

Configuration:
The main configuration file is config.py. Set the dataset via command-line argument or within the config (e.g., --dataset se2018_single).
HypEmO-specific hyperparameters (alpha, gamma, dim, etc.) are also set via config.py or command-line arguments. You might need to tune alpha and gamma specifically for the SE2018 dataset.
Execution: Assuming the main training script is train.py (based on the original README) and it accepts command-line arguments:

```Bash
# Example: Run HypEmO (assuming RoBERTa backbone via config) on se2018_single with seed 111
python train.py --dataset se2018_single --seed 111 # Potentially add --alpha, --gamma etc.
```
Multiple Runs: Similar to baselines, run the HypEmO experiment 5 times with different SEED values for reliable comparison.

### Hyperparameters
Baselines: Key hyperparameters (learning rate, epochs, batch size) are defined in the configuration section of train_evaluate_baseline.py or can be passed as arguments.

### Note
train_label_embedding.py contains the script for training hyperbolic label embeddings for custom datasets.
If you run this script for the se2018_single dataset, it will generate a .bin file in the label_tree folder containing the embeddings.
You can then run the main HypEmO training script (e.g., train.py).
If using provided or pre-existing .bin files, you can skip running train_label_embedding.py.

### Results
Output logs and evaluation results (JSON format) for each run are saved in subdirectories within the exp/ folder (e.g., exp/baselines/roberta-base_seed111_.../).
To report results comparable to the paper, calculate the mean and standard deviation of the primary metrics (Accuracy and Weighted F1) across the 5 runs for each model configuration.