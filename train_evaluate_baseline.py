# train_evaluate_baseline.py

import os
import logging
import json
import importlib.util
from datetime import datetime
import random
import numpy as np
import torch
import argparse

import pandas as pd
from datasets import load_dataset, Dataset, Features, ClassLabel, Value
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    logging as hf_logging,
)
from sklearn.metrics import accuracy_score, f1_score
import wandb

# --- 参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate baseline models for emotion classification.")
    parser.add_argument('--model_name', type=str, default="roberta-base", help='Name of the model to use (e.g., roberta-base, bert-base-uncased)')
    parser.add_argument('--seed', type=int, default=111, help='Random seed for reproducibility')
    parser.add_argument('--dataset_name', type=str, default="se2018_single", help='Dataset identifier (used for constructing paths)') # 加入 dataset 参数，虽然代码里没直接用它构造路径，但与 sweep 匹配

    # --- Data settings --- (这些也可以做成参数，但现在先保持原样)
    parser.add_argument('--text_column', type=str, default="text")
    parser.add_argument('--label_column', type=str, default="label")
    parser.add_argument('--config_file', type=str, default="config_se18.py") # 保持指向正确的配置文件
    # 文件路径可以保持相对路径，假设脚本在项目根目录运行
    parser.add_argument('--train_file_path', type=str, default="data/se2018_single/train.csv")
    parser.add_argument('--valid_file_path', type=str, default="data/se2018_single/valid.csv")
    parser.add_argument('--output_dir_base', type=str, default="exp/baselines")

    # --- Training Hyperparameters ---
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--early_stopping_patience', type=int, default=3)

    args = parser.parse_args()
    return args

args = parse_args() # 解析命令行参数

# --- Configuration ---

# --- 使用解析后的参数 ---
MODEL_NAME = args.model_name
SEED = args.seed
CONFIG_FILE = args.config_file
TRAIN_FILE = args.train_file_path
VALID_FILE = args.valid_file_path
OUTPUT_DIR_BASE = args.output_dir_base
TEXT_COLUMN = args.text_column
LABEL_COLUMN = args.label_column
LEARNING_RATE = args.learning_rate
WEIGHT_DECAY = args.weight_decay
NUM_TRAIN_EPOCHS = args.num_train_epochs
TRAIN_BATCH_SIZE = args.train_batch_size
EVAL_BATCH_SIZE = args.eval_batch_size
MAX_SEQ_LENGTH = args.max_seq_length
EARLY_STOPPING_PATIENCE = args.early_stopping_patience

# --- 初始化 W&B ---
# W&B 会自动从环境变量读取 WANDB_PROJECT，或者你可以在这里指定
# 它也会自动读取 args，将参数记录到 config 中
run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "HypEmo-SE18-Baselines"), # 从环境变量读取，或使用默认项目名
    config=args, # 将所有解析的参数传给 W&B config
    name=f"{MODEL_NAME}_seed{SEED}", # 给这次运行起个名字
    reinit=True # 允许在同一脚本中多次初始化 (虽然 sweep 不会这样)
)

# --- Set Seed for Reproducibility ---
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # If you were using GPU:
    # torch.cuda.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

set_seed(SEED)

# --- Output Directory ---
model_name_sanitized = MODEL_NAME.replace('/', '_')
# 使用 wandb run id 创建唯一的目录，或者保持原来的时间戳方式
# run_output_dir = os.path.join(OUTPUT_DIR_BASE, f"{model_name_sanitized}_seed{SEED}_{run.id}") # 使用 W&B run ID
run_output_dir = os.path.join(OUTPUT_DIR_BASE, f"{model_name_sanitized}_seed{SEED}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") # 或者保持时间戳
log_filename = os.path.join(run_output_dir, "run_log.log")
os.makedirs(run_output_dir, exist_ok=True)

# --- Set up Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
hf_logging.set_verbosity_info()
logger = logging.getLogger(__name__)
logger.info(f"--- Starting Run (via W&B Sweep Agent maybe) ---")
logger.info(f"Arguments: {args}") # 记录所有参数
logger.info(f"Output Directory: {run_output_dir}")
logger.info(f"W&B Run URL: {run.get_url()}") # 打印 W&B 运行链接

# --- Load Label Mapping ---
# (Same as before)
label2id = None
id2label = None
num_labels = 0
try:
    spec = importlib.util.spec_from_file_location("config_module", CONFIG_FILE)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    if hasattr(config_module, 'LABEL2ID'):
        label2id = config_module.LABEL2ID
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)
        logger.info(f"Loaded labels from {CONFIG_FILE}. Num labels: {num_labels}")
    else:
        logger.error(f"LABEL2ID not found in {CONFIG_FILE}. Exiting."); exit()
except Exception as e:
    logger.error(f"Error loading config {CONFIG_FILE}: {e}. Exiting."); exit()

# --- Load Data ---
logger.info("Loading datasets...")
try:
    features = Features({
        TEXT_COLUMN: Value('string'),
        'aug_text': Value('string'), 
        LABEL_COLUMN: ClassLabel(num_classes=num_labels, names=list(label2id.keys()))
    })
    raw_datasets = load_dataset(
        'csv',
        data_files={'train': TRAIN_FILE, 'validation': VALID_FILE},
        features=features
    )
    logger.info(f"Loaded train set: {raw_datasets['train']}")
    logger.info(f"Loaded validation set: {raw_datasets['validation']}")

    # Rename label column if necessary
    if LABEL_COLUMN != 'labels':
         raw_datasets = raw_datasets.rename_column(LABEL_COLUMN, "labels")

except Exception as e:
    logger.error(f"Error loading data files ({TRAIN_FILE}, {VALID_FILE}): {e}. Exiting."); exit()

# --- Preprocessing ---
logger.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(examples[TEXT_COLUMN], truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)

logger.info("Tokenizing datasets...")
try:
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=[TEXT_COLUMN])
    logger.info(f"Columns after tokenization: {tokenized_datasets['train'].column_names}")
except Exception as e:
    logger.error(f"Error during tokenization: {e}. Exiting."); exit()

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- Define Metrics ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    # Using weighted F1 as primary metric, similar to paper [cite: 114, 115]
    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_macro = f1_score(labels, predictions, average='macro')
    return {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro
    }

# --- Load Model ---
logger.info(f"Loading model: {MODEL_NAME}")
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
except Exception as e:
    logger.error(f"Error loading model {MODEL_NAME}: {e}. Exiting."); exit()

# --- Training ---
logger.info("Setting up Trainer...")

training_args = TrainingArguments(
    output_dir=run_output_dir,
    # --- 使用 args 中的参数 ---
    do_train=True,
    num_train_epochs=NUM_TRAIN_EPOCHS, # 使用 args.num_train_epochs
    learning_rate=LEARNING_RATE,       # 使用 args.learning_rate
    weight_decay=WEIGHT_DECAY,         # 使用 args.weight_decay
    per_device_train_batch_size=TRAIN_BATCH_SIZE, # 使用 args.train_batch_size
    warmup_steps=50,

    do_eval=True,
    eval_strategy="epoch",
    per_device_eval_batch_size=EVAL_BATCH_SIZE, # 使用 args.eval_batch_size

    save_strategy="epoch",
    save_total_limit=2,
    logging_dir=os.path.join(run_output_dir, 'logs'),
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,

    seed=SEED, # 传递 seed 给 Trainer
    # report_to="wandb", # <--- 不再需要，wandb.init 已经处理
    report_to="none", # <--- 设置为 none，避免 Trainer 再次尝试初始化
    no_cuda=True,
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
)


# --- Run Training & Evaluation ---
logger.info(f"--- Starting Training & Evaluation for {MODEL_NAME} with Seed {SEED} ---")
logger.info(f"Hyperparameters: LR={LEARNING_RATE}, WeightDecay={WEIGHT_DECAY}, Epochs={NUM_TRAIN_EPOCHS}, TrainBatch={TRAIN_BATCH_SIZE}, EvalBatch={EVAL_BATCH_SIZE}, MaxLen={MAX_SEQ_LENGTH}")
logger.info(f"Running on CPU...")

try:
    train_result = trainer.train()
    logger.info("--- Training Complete ---")
    # Log training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state() # Saves optimizer state etc.

    # The best model is already loaded because load_best_model_at_end=True
    # Evaluate the best model on the validation set
    logger.info("--- Evaluating Best Model on Validation Set ---")
    eval_results = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])

    logger.info("--- Final Evaluation Results (Best Model) ---")
    # Print results cleanly
    for key, value in eval_results.items():
        logger.info(f"{key}: {value:.4f}")

    # Save final evaluation results
    results_filename = os.path.join(run_output_dir, "best_model_eval_results.json")
    with open(results_filename, 'w') as f:
        json.dump(eval_results, f, indent=4)
    logger.info(f"Best model evaluation results saved to {results_filename}")

    # Optional: Save the best model explicitly (Trainer already saves checkpoints)
    # best_model_path = os.path.join(run_output_dir, "best_model")
    # trainer.save_model(best_model_path)
    # logger.info(f"Best model saved to {best_model_path}")

except Exception as e:
    logger.error(f"An error occurred during training or evaluation: {e}", exc_info=True) # Log traceback
# --- 结束 W&B Run ---
run.finish() # 明确结束 W&B run
logger.info(f"Script finished for Seed {SEED}. Main log: {log_filename}")