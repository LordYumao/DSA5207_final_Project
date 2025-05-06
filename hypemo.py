import os
import json
import time
import pickle
import logging
import datetime
import optimizers
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score


from config import parser
from models.base_models import FGTCModel
from util_functions import *
from util_functions import HyoEmoDataSet
from hypbert import HypBert
from transformers import AutoConfig

args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


class HypEmo():
    def __init__(self, args, n_classes, class_names, idx2vec, alpha, gamma,
                 train_loader=None, valid_loader=None, test_loader=None): # 将 loaders 设为可选参数

        self.args = args
        self.device = self.args.device # 从 args 获取设备
        self.class_names = class_names # 保存其他传入的参数
        self.idx2vec = idx2vec

        # ================== START: 条件 Loader 逻辑 ==================
        if train_loader is not None and valid_loader is not None and test_loader is not None:
            # --- 情况 1: 外部传入了 Loaders (来自 train_kfold.py) ---
            logging.info("HypEmo initialized using provided DataLoaders.")
            self.train_loader = train_loader
            self.valid_loader = valid_loader
            self.test_loader = test_loader
            # 注意：如果 FGTCModel 等需要 n_samples，可能需要从 loader 获取
            # if hasattr(self.args, 'n_samples') and self.args.n_samples is None:
            #     try:
            #         self.args.n_samples = len(self.train_loader.dataset)
            #         logging.info(f"Set args.n_samples from train_loader: {self.args.n_samples}")
            #     except Exception as e:
            #         logging.warning(f"Could not determine n_samples from provided train_loader: {e}")

        else:
            # --- 情况 2: 没有传入 Loaders (来自 train.py) ---
            logging.info("DataLoaders not provided, creating internally based on args.dataset and args.batch_size.")
            # 确保 args 包含必要信息
            if not hasattr(args, 'dataset') or not hasattr(args, 'batch_size'):
                 raise ValueError("When DataLoaders are not provided, 'args' must contain 'dataset' and 'batch_size'.")

            # --- 执行原始的内部 Loader 创建逻辑 ---
            logging.info(f"Creating internal DataLoaders for dataset '{args.dataset}' with batch_size {args.batch_size}...")
            try:
                # 确保 HyoEmoDataSet 类可用
                trainset = HyoEmoDataSet(args.dataset, 'train')
                # 注意 batch_size 的来源，应该来自 args
                self.train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=trainset.collate)

                validset = HyoEmoDataSet(args.dataset, 'valid')
                # 验证和测试的 batch size 可能需要调整，或者也从 args 获取
                valid_batch_size = getattr(args, 'valid_batch_size', args.batch_size * 2) # 示例：允许不同的验证批大小
                self.valid_loader = DataLoader(validset, batch_size=valid_batch_size, shuffle=False, collate_fn=validset.collate)

                testset = HyoEmoDataSet(args.dataset, 'test')
                test_batch_size = getattr(args, 'test_batch_size', args.batch_size * 2) # 示例：允许不同的测试批大小
                self.test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, collate_fn=testset.collate)

                # 原始的 n_samples 设置逻辑，只在这里执行
                self.args.n_samples = len(trainset)
                logging.info(f"Internal DataLoaders created. Train samples: {self.args.n_samples}")

            except Exception as e:
                 logging.error(f"Failed to create internal DataLoaders: {e}")
                 raise # 重新抛出错误，因为没有 Loader 无法继续

        # =================== END: 条件 Loader 逻辑 ===================
        # 初始化模型和优化器
        args.n_samples, args.feat_dim = len(trainset), 768
        args.n_classes = n_classes
        self.poincare_model = FGTCModel(args)
        self.poincare_optimizer = getattr(optimizers, 'RiemannianAdam')(params=self.poincare_model.parameters(), lr=0.01)
        self.poincare_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.poincare_optimizer, step_size=10, gamma=0.5)
        self.class_names = class_names
        self.idx2vec = idx2vec
        self.model = HypBert(num_labels=args.n_classes, alpha=alpha, gamma=gamma)
        self.model.to(args.device)
        self.poincare_model.to(args.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=1e-2)
        
    def train_step(self, ith_epoch):

        self.model.train()

        train_pred, train_label = [], []
        step = 0
        total_loss, total_poincare_loss = 0.0, 0.0
        p_bar = tqdm(self.train_loader, total=len(self.train_loader))
        
        out_train_X, out_train_y = [], []
        for x, label in p_bar:
            step += 1
            input_ids, attention_mask, labels = x['input_ids'].to(args.device), x['attention_mask'].to(args.device), label.to(args.device)
           
            output = self.model(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                labels=labels, 
                                poincare_model=self.poincare_model,
                                idx2vec=self.idx2vec)
            
            out_train_X.extend(output['cls'].cpu().detach().numpy())
            out_train_y.extend(labels.cpu().detach().numpy())
                
            loss = output['total_loss']
            poincare_loss = output['poincare_loss']

            self.opt.zero_grad()
            self.poincare_optimizer.zero_grad()

            loss.backward()

            self.opt.step()
            self.poincare_optimizer.step()
            self.poincare_lr_scheduler.step()

            total_loss += loss.item()
            total_poincare_loss += poincare_loss.item()

            train_pred.extend(torch.argmax(output['logits'], dim=-1).tolist())
            train_label.extend(label.tolist())

            if step % 10 == 0:
                p_bar.set_description(f'train step {step} | loss={(total_loss/step):.4f}')

        train_acc = accuracy_score(train_pred, train_label)
        train_weighted_f1 = f1_score(train_pred, train_label, average='weighted')
        logging.info(f'''train | loss: {total_loss/step:.04f} acc: {train_acc:.04f}, f1: {train_weighted_f1:.04f}''')
        
        return {'loss': total_loss/step, 'train_acc': train_acc, 'train_weighted_f1': train_weighted_f1}


    def valid_step(self, ith_epoch):
        valid_pred = None
        valid_label = []
        with torch.no_grad():
            for x, label in self.valid_loader:
                input_ids, attention_mask, labels = x['input_ids'].to(args.device), x['attention_mask'].to(args.device), label.to(args.device)
                output = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels, 
                                    poincare_model=self.poincare_model,
                                    idx2vec=self.idx2vec)
                logits = output['logits']
                loss = output['total_loss']
                prediction = torch.argmax(logits, dim=-1)
                if valid_pred is None:
                    valid_pred = prediction
                else:
                    valid_pred = torch.cat([valid_pred, prediction])
                valid_label.extend(label.tolist())

        valid_pred = valid_pred.detach().cpu().numpy()
        valid_acc = accuracy_score(valid_pred, valid_label)
        valid_weighted_f1 = f1_score(valid_pred, valid_label, average='weighted')
        logging.info(f'''valid | loss: {loss:.04f} acc: {valid_acc:.04f}, f1: {valid_weighted_f1:.04f}''')
        return {'valid_loss': loss, 'valid_pred': valid_pred, 'valid_acc': valid_acc, 'valid_weighted_f1': valid_weighted_f1}
    
    def test_step(self, ith_epoch):
        test_pred = None
        test_label = []
        with torch.no_grad():
            for x, label in self.test_loader:
                input_ids, attention_mask = x['input_ids'].to(args.device), x['attention_mask'].to(args.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)['logits']
                prediction = torch.argmax(logits, dim=-1)
                if test_pred is None:
                    test_pred = prediction
                else:
                    test_pred = torch.cat([test_pred, prediction])
                test_label.extend(label.tolist())

        test_pred = test_pred.detach().cpu().numpy()
        test_acc = accuracy_score(test_pred, test_label)
        test_weighted_f1 = f1_score(test_pred, test_label, average='weighted')

        logging.info(f'''test | acc: {test_acc:.04f}, f1: {test_weighted_f1:.04f}''')
        return {'test_pred': test_pred, 'test_acc': test_acc, 'test_weighted_f1': test_weighted_f1}