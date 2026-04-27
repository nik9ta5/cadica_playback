#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
from pathlib import Path
from tqdm import tqdm
import datetime as dt
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models 
from torchvision.models import MobileNet_V2_Weights

from sklearn.model_selection import StratifiedGroupKFold


# LCA, эксперимент, который выполнялся при определении параметров.
# 
# Model: `MobileNet_v2`
# 
# Тут возьмем другую скорость обучения
# 
# Parameters:
# - Batch_size: 16
# - Optimizer: RMSProp
# - LR: 1e-5
# 
# Полученные метрики (из статьи):
# * F1:
# * Balanced:
# * Accuracy:
# 
# Полученные в результате запуска
# * f1:
# * balanced_accuracy:
# * accuracy:

# In[2]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 16
LR = 1e-5
EPOCHS = 10

CV_N = 5
STEP_BATCH_LOG = 50

input_img_shape = (3,224,224)
split_level = "video_id"


path2model_save = Path("../models_checkpoints")
os.makedirs(path2model_save, exist_ok=True)

model_name = "mobilenet_v2_backbone_cv" # model train name
prefix_train = "full_tuning_video" # prefix train


comment = f""" 
===== general parameters =====
SEED: {SEED}
DEVICE: {DEVICE}

BATCH_SIZE: {BATCH_SIZE}
input_img_shape: {input_img_shape}
split_level: {split_level}

===== train parameters =====

LR: {LR}


CV_N: {CV_N}
EPOCHS: {EPOCHS}
STEP_BATCH_LOG: {STEP_BATCH_LOG}

===== about model and save =====
base_dir_for_model_save: {path2model_save}

model_name: {model_name}
prefix_train: {prefix_train}

===== comments =====
LCA
cross-validation
split patient level
SGDM optimizer
use augmentation
use ImageNet stats for (mean, std) normalization 
"""

# ### Загрузка датасета

# In[3]:


df_nonleison_metadata = pd.read_csv("../data/lca_nonleison_metadata_aug.csv")
df_leison_metadata = pd.read_csv("../data/lca_leison_metadata_aug.csv")

df_test_metadeta = pd.read_csv("../data/lca_test.csv")

# === Объединяем фреймы (Нужно перемешать) ===
df_train = pd.concat([df_nonleison_metadata, df_leison_metadata], ignore_index=True)

# ### Model architecture - MobileNet_v2

# In[4]:


class MobileNet_v2_Backbone(nn.Module):
    def __init__(self, img_size: tuple[int, int, int], out_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.in_channels = img_size[0]
        self.img_size = img_size
        self.out_classes = out_classes
        self.freeze_backbone = freeze_backbone

        # === init ===
        self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # === change classifier layer ===
        in_features_classifier = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features_classifier, out_features=self.out_classes)
        
        # === freeze backbone ===
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        return out

# ### Dataset

# In[5]:


class CADICADataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_path_col: str, label_col: str, transform = None):
        self.dataframe = dataframe
        self.img_path_col = img_path_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        path_to_img = self.dataframe.loc[idx, self.img_path_col]
        image = Image.open(path_to_img).convert("RGB")
        label = self.dataframe.loc[idx, self.label_col]
        if self.transform:
            image = self.transform(image)
        return image, label

# In[6]:


def create_logger(dir_for_save: str, log_file: str):
    
    full_path2save = f"{dir_for_save}/{log_file}"

    logger = logging.getLogger("ModelTrainer")
    logger.setLevel(logging.INFO) 

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(full_path2save, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) 
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# In[7]:


def save_checkpoint(state: dict, dir_name: str, filename: str) -> str:
    path2save = f"{dir_name}/{filename}"
    torch.save(state, path2save)
    return path2save

def compute_metrics(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: str = "cpu"
    ):
    """Считает все важные метрики на валидационном/тестовом датасете"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Computing metrics"):

            outputs = model(images.to(device))
            preds = torch.argmax(outputs, dim=1)          
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="binary", pos_label=1),   # lesion = 1
        "precision": precision_score(all_labels, all_preds, average="binary", pos_label=1),
        "recall": recall_score(all_labels, all_preds, average="binary", pos_label=1),
    }
    
    return metrics

def valid_loop(
    model: nn.Module, 
    loss_fn, 
    dataloader: DataLoader, 
    device: str = "cpu"
    ):
    """Функция с циклом валидации"""
    model.eval()
    total_loss = 0.0
    num_batchs = 0
    for batch_idx, (images, labels) in tqdm(enumerate(dataloader), desc="valid"):
        with torch.no_grad():
            out = model(images.to(device))
        loss = loss_fn(out, labels.to(device))      
        total_loss += loss.item()
        num_batchs+=1
    total_loss /= num_batchs
    return total_loss

def befor_train_init(
    dir_for_model_save: str,
    model_name: str,
    time_start: str,
    prefix_train: str
    ):
    # === create base dir for checkpoints and logs ===
    os.makedirs(dir_for_model_save, exist_ok=True)
    
    # === create dir for arch model ===
    os.makedirs(f"{dir_for_model_save}/{model_name}", exist_ok=True)

    # === create dir for current experiment ====
    dir_name = f"{time_start}_{prefix_train}"
    full_path_dir = f"{dir_for_model_save}/{model_name}/{dir_name}"
    os.makedirs(full_path_dir, exist_ok=True)

    # checkpoint dir
    os.makedirs(f"{full_path_dir}/checkpoint", exist_ok=True)
    return full_path_dir

def train_loop(
    model: nn.Module, 
    loss_fn, 
    optimizer, 
    dataloader: DataLoader, 
    valid_loader: DataLoader, 
    path_to_model_save: str,
    step_batch_log: int = 10,
    early_stopping_epoch: int = 0,
    logger = None,
    device: str = "cpu", 
    epochs: int = 1,
    scheduler = None,
    seed: int = 42
    ):

    history = {
        "totals_loss" : [],
        "valids_loss" : [],
        "balanced_accuracies" : [],
        "metrics": []
    }
    
    best_balanced_accuracy = 0.0
    early_counter = 0

    for epoch in range(epochs):
        
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in tqdm(enumerate(dataloader), desc="train"):
            optimizer.zero_grad(set_to_none=True) # <--- CHANGE (set_to_none=True)
            # images: torch.Size([16, 3, 224, 224])
            out = model(images.to(device))
            loss = loss_fn(out, labels.to(device))      
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (batch_idx+1)%step_batch_log==0:
                if logger:
                    num_exmpls = images.shape[0] * (batch_idx+1)
                    logger.info(f"epoch {epoch+1} | batch {batch_idx+1} | train_loss: {(total_loss/num_exmpls):.8f}")

        # ===========================================================================

        total_loss /= len(dataloader)
        
        # === valid ===
        valid_loss = valid_loop(model, loss_fn, valid_loader, device=device)
        
        # === metrics calculate ===
        metrics = compute_metrics(model, valid_loader, device)
        curr_balanced_accuracy = metrics.get("balanced_accuracy", 0.0)
        
        # === save for log ===
        history['totals_loss'].append(total_loss)
        history['valids_loss'].append(valid_loss)

        for key in list(metrics.keys()):
            if key not in history:
                history[key] = []
            history[key].append(metrics[key])


        # === scheduler run ===
        if scheduler:
            scheduler.step(curr_balanced_accuracy)

        # === logging ===
        if logger:
            logger.info(f"================= epoch end =================")
            logger.info(f"epoch {epoch+1} | train_loss: {total_loss:.8f} | valid_loss: {valid_loss:.8f} | valid balanc. acc.: {curr_balanced_accuracy:.8f}")
            # log metrics
            logger.info(f"metrics:\n{metrics}")

        # === save best model ===
        if curr_balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = curr_balanced_accuracy
            early_counter = 0
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_blns_acc": best_balanced_accuracy,
                "seed": seed
            }
            path2save = save_checkpoint(state, f"{path_to_model_save}/checkpoint", "best_model.pth")
            if logger:
                logger.info(f"save checkpoint: {path2save}")
        else:
            early_counter += 1

        # === create early stopping ===
        if (early_stopping_epoch != 0) and (early_counter >= early_stopping_epoch):
            if logger:
                logger.info(f"Early stopping. Epoch: {epoch+1}")
                logger.info(f"best balanced accuracy: {best_balanced_accuracy}")
            break
        
    return history

# In[8]:


time_now = dt.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
path2save = befor_train_init(
    path2model_save,
    model_name,
    time_now,
    prefix_train
)
train_logger = create_logger(path2save, "train_log.log")
train_logger.info(comment)

# In[9]:


transform_images = v2.Compose([
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ### CV

# In[10]:


# Группировка по видео (все фреймы одного видео идут в одну часть)
groups = df_train['video_id'].values
sgkf = StratifiedGroupKFold(n_splits=CV_N, shuffle=True, random_state=SEED)


for fold, (train_idx, val_idx) in enumerate(sgkf.split(df_train, df_train['label'], groups)):
    train_logger.info(f"Fold {fold + 1}")

    # === for dataset for train and val ===
    df_train_folds = df_train.iloc[train_idx].reset_index(drop=True)
    df_val_fold = df_train.iloc[val_idx].reset_index(drop=True)
    
    train_logger.info(f'''
    ======= about folds =======
    {df_train_folds.shape, df_val_fold.shape}
    {set(df_train_folds['video_id'].unique()).intersection(df_val_fold['video_id'].unique())}
    {df_train_folds.groupby("label")['image_path'].count()}     
    {df_val_fold.groupby("label")['image_path'].count()}
    ''')

    # === create dataset objects ===
    df_train_folds = CADICADataset(df_train_folds, "image_path", "label", transform=transform_images)
    df_val_fold = CADICADataset(df_val_fold, "image_path", "label", transform=transform_images)

    # === create loaders ===
    train_loader = DataLoader(df_train_folds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(df_val_fold, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # === init model ===
    model = MobileNet_v2_Backbone(img_size=input_img_shape, out_classes=2, freeze_backbone=False).to(DEVICE)
    optimizer = optim.RMSprop(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_logger.info(f'''
    Всего параметров: {total_params:,}
    Обучаемых:       {trainable_params:,}
    ''')

    # ========== create dir for save model ==========
    path2save_model_fold = f"{path2save}/fold{fold+1}"
    os.makedirs(path2save_model_fold, exist_ok=True)
    os.makedirs(f"{path2save_model_fold}/checkpoint", exist_ok=True)

    # ========== Запуск цикла обучения ==========
    train_history = train_loop(
        model, 
        loss_fn, 
        optimizer, 
        train_loader, 
        val_loader, 
        path_to_model_save=path2save_model_fold,
        step_batch_log = STEP_BATCH_LOG,
        early_stopping_epoch = 0,
        logger = train_logger,
        device = DEVICE, 
        epochs = EPOCHS,
        seed = SEED
    )

    # ========== Сохранение метрик хода обучения ==========
    with open(f"{path2save_model_fold}/train_history.json", "w", encoding="utf-8") as f:
        json.dump(train_history, f, ensure_ascii=False, indent=4)

# ### Metrics visualizate

# In[16]:


metrics_for_plots = dict()

for i in range(CV_N):
    fold = f"fold{i+1}"
    path2metrics_data = f"{path2save}/{fold}/train_history.json"
    with open(path2metrics_data, 'r') as f:
        metrics = json.load(f)
    metrics_for_plots[fold] = metrics

# In[17]:


path2save

# In[22]:


metrics_keys = ["totals_loss", "valids_loss", "accuracy", "balanced_accuracy", "f1", "precision", "recall"]
rows = len(metrics_keys)
cols = 1

fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 8, rows * 4))

for i in range(rows):    

    for fold in list(metrics_for_plots.keys()):
        
        ax[i].plot(metrics_for_plots[fold][metrics_keys[i]], label=fold)

    ax[i].legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax[i].set_title(metrics_keys[i])
    ax[i].grid(True)

plt.tight_layout() 
plt.show()

# In[23]:


# Полученные метрики в ходе обучения

mean_metrics_per_folds = {}

for fold in metrics_for_plots:
    item_accuracy_mean = np.mean(metrics_for_plots[fold]["accuracy"])    
    item_balanced_accuracy_mean = np.mean(metrics_for_plots[fold]["balanced_accuracy"])
    item_f1_mean = np.mean(metrics_for_plots[fold]["f1"])
    
    item_accuracy_std = np.std(metrics_for_plots[fold]["accuracy"])    
    item_balanced_accuracy_std = np.std(metrics_for_plots[fold]["balanced_accuracy"])
    item_f1_std = np.std(metrics_for_plots[fold]["f1"])

    mean_metrics_per_folds[fold] = {
        "accuracy": item_accuracy_mean,
        "balanced_accuracy": item_balanced_accuracy_mean,
        "f1": item_f1_mean,   
    }

    print(f'''{fold}
accuracy: {item_accuracy_mean:.3f} +- {item_accuracy_std:.3f}
balanced_accuracy: {item_balanced_accuracy_mean:.3f} +- {item_balanced_accuracy_std:.3f}
f1: {item_f1_mean:.3f} +- {item_f1_std:.3f}
''')


# In[24]:


# === Финальные метрики ===
for metric in ['accuracy', 'balanced_accuracy', 'f1']:
    values = [mean_metrics_per_folds[f][metric] for f in mean_metrics_per_folds.keys()]
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")
