# Reproduction of the results presented in the paper "CADICA: a new dataset for coronary artery disease detection using invasive coronary angiography."

[link to the article](https://arxiv.org/pdf/2402.00570)

Repository structure:
* `models_checkpoints/` - Catalog of experiment logs
* `notebooks/` - Notebooks with learning pipelines

Data preparation notebooks:
* `notebooks/dataframe_create.ipynb` - Notebook with creation of a dataframe with metainformation about the dataset
* `notebooks/cadica_visualization.ipynb` - Cadica frame visualization 
* `notebooks/dataframe_lca_preprocess.ipynb` - Image preprocessing before training

## Результаты

1. `notebooks/baseline_LCA_ResNet18.ipynb` - Первый эксперимент (с воспроизведением эксперимента, при поиске параметров) 

Файл лога: `./models_checkpoints/resnet_18_backbone_cv/2026-04-23_18_17_44_full_tuning_patient` 

LCA, ResNet-18, 16, 1e-4, RMSProp, Валидация в конце эпохи

Статья:
* F1: 0.800 +- 0.021
* Balanced Accuracy: 0.623 +- 0.017
* Accuracy: 0.708 +- 0.023

Полученный результат:
* F1: 0.650 ± 0.042
* Balanced Accuracy: 0.606 ± 0.017
* Accuracy: 0.603 ± 0.036


2. `notebooks/baseline_LCA_ResNet18_v2.ipynb` - Изменена learning rate

Файл лога: `./models_checkpoints/resnet_18_backbone_cv/2026-04-23_21_34_14_full_tuning_video`

LCA, ResNet-18, 16, 1e-5, RMSProp, Валидация в конце эпохи

Статья:
* F1: 0.789 ± 0.019
* Balanced Accuracy: 0.649 ± 0.024
* Accuracy: 0.705 ± 0.022

Полученный результат:
* F1: 0.667 ± 0.055
* Balanced Accuracy: 0.625 ± 0.031
* Accuracy: 0.621 ± 0.049


3. `notebooks/baseline_LCA_ResNet18_v3.ipynb`

Файл лога: `./models_checkpoints/resnet_18_backbone_cv/2026-04-24_00_06_06_full_tuning_video`

LCA, ResNet-18, 16, 1e-5, RMSProp, Валидация каждые 50 итераций. 

Статья:
* F1: 0.789 ± 0.019
* Balanced Accuracy: 0.649 ± 0.024
* Accuracy: 0.705 ± 0.022

Полученный результат:
* F1: 0.658 ± 0.048
* Balanced Accuracy: 0.609 ± 0.022
* Accuracy: 0.605 ± 0.042