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

SEED = 42

np.random.seed(SEED)

# ## Формирование датасета

# In[2]:


# Директория с отобранными видео пациентов
PATH_SELECTED_VIDEO = Path("../../datasets/CADICA/selectedVideos")

# Файл с описанием какое видео относится к LCA и RCA
CADICAprojections_path = "../../datasets/CADICA/CADICAprojections.json"

# In[3]:


def read_file(path: str):
    """Функция для считывания файла .txt"""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# In[4]:


records = []

all_patients = sorted([p for p in os.listdir(PATH_SELECTED_VIDEO) if os.path.isdir(PATH_SELECTED_VIDEO / p)])

print(f"Найдено пациентов: {len(all_patients)}")

for px in tqdm(all_patients, desc="Обработка пациентов"):
    patient_path = PATH_SELECTED_VIDEO / px
    
    # Читаем списки видео
    lesion_videos = read_file(patient_path / "lesionVideos.txt")
    nonlesion_videos = read_file(patient_path / "nonlesionVideos.txt")
    
    # === Обработка видео с поражениями (label = 1) ===
    for vx in lesion_videos:
        video_path = patient_path / vx
        frames_txt = video_path / f"{px}_{vx}_selectedFrames.txt"
        gt_dir = video_path / "groundtruth"
        
        selected_frames = read_file(frames_txt)
        
        for frame_name in selected_frames:
            img_path = video_path / "input" / f"{frame_name}.png"
            gt_path = gt_dir / f"{frame_name}.txt"
            
            if not img_path.exists():
                continue
                
            # Проверяем, действительно ли есть поражение
            has_lesion = False
            num_bboxes = 0
            if gt_path.exists():
                bboxes = read_file(gt_path)
                num_bboxes = len(bboxes)
                has_lesion = num_bboxes > 0
            
            records.append({
                'patient': px,
                'video': vx,
                'frame': frame_name,
                'image_path': str(img_path),
                'label': 1 if has_lesion else 0,
                'num_bboxes': num_bboxes
            })
    
    # === Обработка видео без поражений (label = 0) ===
    for vx in nonlesion_videos:
        video_path = patient_path / vx
        frames_txt = video_path / f"{px}_{vx}_selectedFrames.txt"
        
        selected_frames = read_file(frames_txt)
        
        for frame_name in selected_frames:
            img_path = video_path / "input" / f"{frame_name}.png"
            
            if not img_path.exists():
                continue
                
            records.append({
                'patient': px,
                'video': vx,
                'frame': frame_name,
                'image_path': str(img_path),
                'label': 0,
                'num_bboxes': 0
            })

# ====================== Создание DataFrame ======================
df = pd.DataFrame(records)

# In[5]:


# Добавить id для видео
df["video_id"] = df['patient'] + "_" + df["video"]

# === add LCA or RCA informations ===
with open(CADICAprojections_path, "r") as file:
    rca_lca_data = json.load(file)

# rca_lca_data['videosLCA2'] - еще есть около 30 видео с LCA
lca_videos = set(rca_lca_data['videosLCA'])
rca_videos = set(rca_lca_data["videosRCA"])

df["artery"] = df["video_id"].apply(lambda x: "RCA" if x in rca_videos else ("LCA" if x in lca_videos else "Unknown"))

print(df.shape)
print(df.groupby("artery")["video_id"].nunique())

# In[6]:


df.groupby('artery').agg({
    "label" : "count",
    "video_id" : "nunique"
})

# ## Сохраняем весь датасет

# In[7]:


df.to_csv("../data/cadica_meta_dataset.csv", index=False)

# ### LCA save

# In[8]:


df_lca = df[df['artery'] == "LCA"]
df_lca.to_csv("../data/cadica_meta_dataset_lca.csv", index=False)

df_lca.groupby('label')["frame"].count()

# ### RCA save

# In[9]:


df_rca = df[df['artery'] == "RCA"]
df_rca.groupby('label')["frame"].count()

df_rca.to_csv("../data/cadica_meta_dataset_rca.csv", index=False)
