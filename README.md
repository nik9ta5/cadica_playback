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

---
4. `./notebooks/old_experiments/train_baseline_ResNet_18.ipynb`

LCA. Сплит на уровне видео

Модель: ResNet-18

Параметры:
SEED: 42
batch_size: 64
LR: 1e-3
optimizer: SGDM
epochs: 5

Momentum: 0.9
Lambda_LR: 1e-4
class_weights = [2.2, 1]
early_stopping: 5

Loss function: CrossEntropyLoss

Предобработка изображения: Использовались статистики для нормализации изображений (ImageNet)

Без кросс-валидации
Датасет разбился на train и test
Потом train разбился на train и val

Датасет не расширялся до значений заявленных в статье (LCA: 3640, RCA: 2342), применялись трансформации к train части датасета.

2 стадии обучения:
1) Заменяем последний слой - обучаем с большим LR (1e-3)
2) Берем лучший checkpoint, размораживаем всю сеть и обучаем с маленьким LR (1e-5)

checkpoint после первого этапа:
`../models_checkpoints/resnet_18_backbone/2026-04-09_00_18_33_classifier_layer_train/checkpoint/best_model.pth`

Новый LR: 1e-5

Оценка лучшего checkpointa на test set не производилась (не после 1-ого не после 2-ого этапов).
Файл лога есть. 
Сохраненных метрик нет, только в логах.
Есть веса модели. Можно написать скрипт и получить оценку на тестовой части.

5. `./notebooks/old_experiments/train_baseline_ResNet_18_version_2.ipynb`

LCA
Сплит на уровне видео

Модель: ResNet-18

Параметры:
SEED: 42
batch_size: 64
LR: 1e-3
optimizer: SGDM
epochs: 20

Momentum: 0.9
Lambda_LR: 1e-4
class_weights = [2.2, 1]
early_stopping: 5

Loss function: CrossEntropyLoss

Использовались статистики для нормализации изображений (ImageNet)

Без кросс-валидации
Датасет разбился на train и test
Потом train разбился на train и val

Датасет не расширялся до значений заявленных в статье (LCA: 3640, RCA: 2342), применялись трансформации к train части датасета.

2 стадии обучения:
1) Заменяем последний слой - обучаем с большим LR (1e-3)
2) Берем лучший checkpoint, размораживаем всю сеть и обучаем с маленьким LR (1e-5)

checkpoint после первого этапа:
../models_checkpoints/resnet_18_backbone/2026-04-09_01_07_22_classifier_layer_train_v2/checkpoint/best_model.pth


Новый LR: 1e-5

Второй этап не проводился
Оценка лучшего checkpointa на test set не производилась (не после 1-ого не после 2-ого этапов).
Файл лога есть. 
Сохраненных метрик нет, только в логах.

6. `./notebooks/old_experiments/train_baseline_ResNet_18_version_3.ipynb`

LCA
Сплит на уровне видео

Модель: ResNet-18

Параметры:
SEED: 42
batch_size: 64
LR: 1e-3
optimizer: SGDM
epochs: 20

Momentum: 0.9
Lambda_LR: 1e-4
class_weights = [1.7, 1]
early_stopping: 5

Loss function: CrossEntropyLoss

Использовались статистики для нормализации изображений (ImageNet)

Без кросс-валидации
Датасет разбился на train и test
Потом train разбился на train и val

Датасет не расширялся до значений заявленных в статье (LCA: 3640, RCA: 2342), применялись трансформации к train части датасета.

Один этап обучения.
Размораживалась сразу вся сеть

checkpoint после первого этапа:
../models_checkpoints/resnet_18_backbone/2026-04-09_13_20_39_classifier_layer_train_v2/checkpoint/best_model.pth

Новый LR: 1e-5

Оценка лучшего checkpoint проводилась на test set

{'accuracy': 0.6062874251497006, 'balanced_accuracy': 0.5019463255580637, 'f1': 0.7346115035317861, 'precision': 0.6642335766423357, 'recall': 0.8216704288939052}

Файл лога есть. 
Сохраненных метрик нет, только в логах.
Есть веса модели. Можно написать скрипт и получить оценку на тестовой части.

7. `./notebooks/old_experiments/train_baseline_ResNet_18_version_4_cv.ipynb`

LCA
Сплит на уровне видео

Модель: ResNet-18

Параметры:
SEED: 42
batch_size: 64
LR: 1e-3
optimizer: SGDM
epochs: 10

Momentum: 0.9
Lambda_LR: 1e-4
class_weights = [1.7, 1]
early_stopping: 5

Loss function: CrossEntropyLoss

Использовались статистики для нормализации изображений (ImageNet)

C кросс-валидацией
Датасет разбился на train и test
Потом train разбился на train и val

Датасет не расширялся до значений заявленных в статье (LCA: 3640, RCA: 2342), применялись трансформации к train части датасета.

Один этап обучения.
Размораживалась сразу вся сеть

Все фолды сохранены:
`../models_checkpoints/resnet_18_backbone_cv/2026-04-09_18_41_57_full_tuning_patient/`


Проводилась оценка всех моделей на test set

* accuracy_score: 0.60628743
* balanced_accuracy_score: 0.53147228
* f1_score: 0.71931697
* precision_score: 0.68218623
* recall_score: 0.76072235

Оценка каждой отдельной модели на test set
```
fold1 | accuracy: 0.6302 | balanced_accuracy: 0.5309 | f1: 0.7497 | precision: 0.6801 | recall: 0.8352 | 
fold2 | accuracy: 0.5883 | balanced_accuracy: 0.5485 | f1: 0.6835 | precision: 0.6972 | recall: 0.6704 | 
fold3 | accuracy: 0.6213 | balanced_accuracy: 0.5220 | f1: 0.7431 | precision: 0.6753 | recall: 0.8262 | 
fold4 | accuracy: 0.6347 | balanced_accuracy: 0.5682 | f1: 0.7371 | precision: 0.7052 | recall: 0.7720 | 
fold5 | accuracy: 0.5823 | balanced_accuracy: 0.5462 | f1: 0.6760 | precision: 0.6962 | recall: 0.6569 |
```


Файл лога есть. 
Есть сохраненные метрики для каждого фолда

8. `./notebooks/old_experiments/train_baseline_MobileNet_v2_version_1_cv.ipynb`

LCA
Сплит на уровне видео

Модель: MobileNet-v2

Параметры:
SEED: 42
batch_size: 64
LR: 1e-5
optimizer: SGDM
epochs: 10

Momentum: 0.9
Lambda_LR: 1e-4
class_weights = [1.7, 1]
early_stopping: 5

Loss function: CrossEntropyLoss

Использовались статистики для нормализации изображений (ImageNet)

C кросс-валидацией
Датасет разбился на train и test
Потом train разбился на train и val

Датасет не расширялся до значений заявленных в статье (LCA: 3640, RCA: 2342), применялись трансформации к train части датасета.

Один этап обучения.
Размораживалась сразу вся сеть

Все фолды сохранены:
`../models_checkpoints/mobilenet_v2_backbone_cv/2026-04-10_11_15_57_full_tuning_video_split/`


Проводилась оценка всех моделей на test set

* accuracy_score: 0.62574850
* balanced_accuracy_score: 0.53411588
* f1_score: 0.74279835
* precision_score: 0.68241966
* recall_score: 0.81489842

Оценка каждой отдельной модели на test set
```
fold1 | accuracy: 0.6452 | balanced_accuracy: 0.4952 | f1: 0.7812 | precision: 0.6609 | recall: 0.9549 | 
fold2 | accuracy: 0.5599 | balanced_accuracy: 0.5140 | f1: 0.6636 | precision: 0.6729 | recall: 0.6546 | 
fold3 | accuracy: 0.6347 | balanced_accuracy: 0.5671 | f1: 0.7376 | precision: 0.7043 | recall: 0.7743 | 
```

Файл лога есть. 
Есть сохраненные метрики для каждого фолда

---

## Визуализация

Блокноты для визуализации датасета:
* `./notebooks/dataset_visualization.ipynb`
* `./notebooks/cadica_visualization.ipynb`

### Примеры видео сосудов с поражениями

#### Пример видео (p12, v7)
![Гифка](output_graphics/p12_v7_output_video.gif)

#### Пример видео (p11, v8)
![Гифка](output_graphics/p11_v8_output_video.gif)

#### Пример видео (p11, v16)
![Гифка](output_graphics/p11_v16_output_video.gif)

#### Пример видео (p40, v1)
![Гифка](output_graphics/p40_v1_output_video.gif)


#### Пример фреймов с поражениями (p12, v7)
![Картинка](output_graphics/p12_v7_output_image.png)