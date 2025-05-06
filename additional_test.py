import os
import json
import time
import pickle
import datetime
import optimizers
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

from pytorch_metric_learning import losses
from config import parser, label_dicts, emb_dicts
from sentence_transformers import SentenceTransformer

from util_functions import *
from hypemo import HypEmo

import wandb

args = parser.parse_args()

label2idx, idx2label = label_dicts
num_classes = len(idx2label.items())
class_names = [v for k, v in sorted(idx2label.items(), key=lambda item: item[0])]
word2vec, idx2vec = emb_dicts
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)

gm = HypEmo(args.dataset, num_classes, class_names, idx2vec, args.alpha, args.gamma, batch_size=args.batch_size)
gm.load_model('checkpoint.pt')
gm.pred_step('pred_res.csv')