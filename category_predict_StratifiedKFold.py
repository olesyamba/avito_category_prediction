# Установка необходимых библиотек
# !pip install transformers datasets==2.0.0 scikit-learn torch tensorboard
# pip install transformers[torch]
# Контроль обучения
# tensorboard --logdir=./logs
import os
import pandas as pd
import numpy as np

import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments
from datasets import load_metric
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Подготовка директорий для сохранения чек-поинтов, логов и результатов
root_path = r'C:\Users\olesya.krasnukhina\PycharmProjects\avito_category_predict'
VERSION = 'V1'
n_folds = 3

# Перечень директорий для создания
directories = [os.path.join(root_path, f"model_{VERSION}"),
               os.path.join(root_path, f"model_{VERSION}", "final_logs"),
               os.path.join(root_path, f"model_{VERSION}", "final_model"),
               os.path.join(root_path, f"model_{VERSION}", "final_results")]

for i in range(n_folds):
    directories.append(os.path.join(root_path, f"model_{VERSION}", f"results_fold_{i}"))
    directories.append(os.path.join(root_path, f"model_{VERSION}", f"logs_fold_{i}"))


# Функция для создания недостающих директорий
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")


# Проверка на наличие директории и создание ее, если ее еще нет
for directory in directories:
    create_directory(directory)


# 1. Загрузка данных
df = pd.read_csv('data/train.csv', nrows=500)  # Assuming your data is in a CSV file
rename_dict = {'Category_name' : 'category_name', 'Category' : 'category'}
df.rename(columns=rename_dict, inplace=True)
del rename_dict

df = df[['title', 'category_name']]  # Select relevant columns

# 2. Преобразование категорий в числовой формат
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['category_name'])

# 3. Загрузка токенизатора BERT для русского языка
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')


# Токенизация данных
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt", max_length=128)


# 4. Stratified K-Fold для разделения данных
skf = StratifiedKFold(n_splits=n_folds)

# Метрика для оценки
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Check if input is a NumPy array and convert it
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)


# 5. Класс для создания датасетов
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# 6. Обучение модели с использованием Stratified K-Fold
accuracies = []

for fold, (train_index, test_index) in enumerate(skf.split(df['title'], df['label'])):
    print(f"Fold {fold + 1}")

    # Получаем обучающие и тестовые данные для текущего фолда
    train_texts = df['title'].iloc[train_index].tolist()
    test_texts = df['title'].iloc[test_index].tolist()
    train_labels = df['label'].iloc[train_index].tolist()
    test_labels = df['label'].iloc[test_index].tolist()

    # Токенизация
    train_encodings = tokenize_function(train_texts)
    test_encodings = tokenize_function(test_texts)

    # Создание датасетов для текущего фолда
    train_dataset = TextDataset(train_encodings, train_labels)
    test_dataset = TextDataset(test_encodings, test_labels)

    # Загрузка модели BERT для классификации
    model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased',
                                                          num_labels=len(set(df['label'])))
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    output_dir = os.path.join(root_path, f"model_{VERSION}", f"results_fold_{i}")
    log_dir = os.path.join(root_path, f"model_{VERSION}", f"logs_fold_{i}")

    # Настройка параметров тренировки
    training_args = TrainingArguments(
        output_dir=output_dir,               # Папка для сохранения моделей
        num_train_epochs=3,                  # Количество эпох
        per_device_train_batch_size=32,      # Размер батча для обучения # 8
        per_device_eval_batch_size=16,       # Размер батча для оценки
        warmup_steps=500,                    # Линейный шаг разогрева
        weight_decay=0.01,                   # Регуляризация
        logging_dir=log_dir,                 # Папка для логов
        logging_steps=10,                    # Шаг логирования
        eval_strategy="epoch",               # Оценка на каждой эпохе
        save_strategy="epoch",               # Сохранение на каждой эпохе
        load_best_model_at_end=True          # Загрузка лучшей модели в конце
    )

    # Создание объекта Trainer для обучения
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # Метрики
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Ранняя остановка
    )

    # Обучение модели
    trainer.train()

    # Оценка на тестовых данных для текущего фолда
    results = trainer.evaluate()
    accuracy = results['eval_accuracy']
    accuracies.append(accuracy)
    print(f"Accuracy for fold {fold + 1}: {accuracy}")

# Финальные результаты по всем фолдам
average_accuracy = sum(accuracies) / len(accuracies)
print(f"Средняя точность (Accuracy) по {len(accuracies)} фолдам: {average_accuracy}")


# Обучение на всем наборе данных

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['title'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)
#
# train_encodings = tokenize_function(df['title'].tolist())
# train_labels = df['label'].tolist()
#
# train_dataset = TextDataset(train_encodings, train_labels)


# Инициализация модели для финального обучения
final_model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased',
                                                            num_labels=len(set(df['label'])))

# Применение метода .contiguous() для финальной модели
for param in final_model.parameters():
    if not param.is_contiguous():
        param.data = param.data.contiguous()

output_dir = os.path.join(root_path, f"model_{VERSION}", f"final_results")
log_dir = os.path.join(root_path, f"model_{VERSION}", f"final_logs")

# Настройка параметров тренировки
final_training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=log_dir,
    logging_steps=10,
    eval_strategy="steps",         # Оценка на каждой эпохе
    save_strategy="steps",
    load_best_model_at_end=True
)

# Обучение финальной модели на всех данных
final_trainer = Trainer(
    model=final_model,
    args=final_training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # Метрики
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Ранняя остановка
)

final_trainer.train()

model_dir = os.path.join(root_path, f"model_{VERSION}", f"final_model")

# Сохранение финальной модели
final_model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

# Предсказание категорий
def predict_category(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    predicted_category = label_encoder.inverse_transform(predictions.cpu().numpy())[0]
    return predicted_category

predict_category("Samsung Galaxy S21")