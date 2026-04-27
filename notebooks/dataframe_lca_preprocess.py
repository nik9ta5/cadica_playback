#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from sklearn.model_selection import train_test_split
    
from torchvision.transforms import v2


SEED = 42
np.random.seed(SEED)

# In[2]:


lca_path_data = Path("../data/cadica_meta_dataset_lca.csv")
lca_frame = pd.read_csv(lca_path_data)

print(lca_frame.shape)
print(lca_frame.groupby("label")['frame'].count())

# Кол-во видео с поражениями и без
lca_frame.groupby('label')['video_id'].nunique()

# ### train/test split
# 
# Разбиение осуществлятеся на уровне видео (т.е кадры одного видео присутствуют только в train или test, чтобы не допустить утечки данных)

# In[3]:


video_meta = lca_frame.groupby('video_id')['label'].first().reset_index()

train_vids, test_vids = train_test_split(
    video_meta['video_id'], 
    test_size=0.2, 
    stratify=video_meta['label'],
    random_state=SEED
)

train_df = lca_frame[lca_frame['video_id'].isin(train_vids)]
test_df = lca_frame[lca_frame['video_id'].isin(test_vids)]


print(train_df.groupby('label')['video_id'].nunique())
print(test_df.groupby('label')['video_id'].nunique())

print(train_df.groupby("label")['frame'].count())
print(test_df.groupby("label")['frame'].count())

# Проверяем на пересечение
print(set(train_df["video_id"].unique()).intersection(set(test_df["video_id"].unique())))

# ### Расширение train 
# Расширить train до 3640 изображений (для 0 и для 1)

# In[4]:


# === create dir for save ===
augmtnt_path_data = Path("../data/augment/lca/train")
path2_nonleison_up = augmtnt_path_data / "no_leison_img"
path2_leison_up = augmtnt_path_data / "leison_img"

os.makedirs(augmtnt_path_data, exist_ok=True)
os.makedirs(path2_nonleison_up, exist_ok=True)
os.makedirs(path2_leison_up, exist_ok=True)


# === transform for original image ===
transform_origin = v2.Compose([
    v2.Resize((224, 224))
])

# === transform for augment image ===
transform_augment = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomAffine(
        degrees=(-25, 25),
        translate=(25/224, 25/224),
        scale=(0.8, 1.7),
        fill=0,
    ),
])


# #### Расширяем без поражений

# In[5]:


def dataframe_iteration(frame: pd.DataFrame, path2save: Path, transform, list_for_metadata: list, postfix: str):
    cnt = 0
    for idx, item in tqdm(frame.iterrows(), desc=f"frame preprocess"):
        
        # open image | transform
        img = transform(Image.open(item["image_path"]).convert("RGB"))
        
        # new path2save
        new_path_to_save = path2save / f'{item["frame"]}_{postfix}_{cnt}.png'
        cnt += 1

        # metadata create
        item_dict = {
            "patient" : item["patient"],
            "video" : item["video"],
            "frame" : item["frame"],
            "image_path" : new_path_to_save,
            "label" : item["label"],
            "num_bboxes" : item["num_bboxes"],
            "video_id" : item["video_id"],
            "artery" : item["artery"]
        }
        
        list_for_metadata.append(item_dict)

        # save img
        img.save(new_path_to_save)


def dataset_augment_create(frame: pd.DataFrame, total_size : int, path2save: Path, transform_orig, transform_augment):

    metadatas_about_frame = []

    # === save orig images === 
    dataframe_iteration(frame, path2save, transform_orig, metadatas_about_frame, "orig")

    diff = total_size - frame.shape[0]
    if diff > 0:
        samples_indexs = np.random.choice(frame.shape[0], diff, replace=True)
        selected_frame = frame.iloc[samples_indexs]

        # === save augment images ===
        dataframe_iteration(selected_frame, path2save, transform_augment, metadatas_about_frame, "augmnt")

    df = pd.DataFrame(metadatas_about_frame)

    return df

# In[6]:


# Расширение до 
total_size = 3640

# ### Предобработка для тренировочных изображений

# In[7]:


nonleison_subset = train_df[train_df['label'] == 0].reset_index(False)
nonleison_subset = nonleison_subset.drop(columns=['index'])

df_nonleison_metadata = dataset_augment_create(nonleison_subset, total_size, path2_nonleison_up, transform_origin, transform_augment)
df_nonleison_metadata.to_csv("../data/lca_nonleison_metadata_aug.csv", index=False)

# In[8]:


leison_subset = train_df[train_df['label'] == 1].reset_index(False)
leison_subset = leison_subset.drop(columns=['index'])

df_leison_metadata = dataset_augment_create(leison_subset, total_size, path2_leison_up, transform_origin, transform_augment)
df_leison_metadata.to_csv("../data/lca_leison_metadata_aug.csv", index=False)

# ### Предобработка для тестовых изображений

# In[ ]:


augmtnt_path_data_test = Path("../data/augment/lca/test")
os.makedirs(augmtnt_path_data_test, exist_ok=True)

df_test_metadata = dataset_augment_create(test_df, test_df.shape[0], augmtnt_path_data_test, transform_origin, transform_augment)
df_test_metadata.to_csv("../data/lca_test.csv", index=False)
