# Установка необходимых библиотек
# !pip install transformers datasets scikit-learn torch
# pip install transformers[torch]
import pandas as pd

import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Загрузка данных
df = pd.read_csv('data/train.csv')  # Assuming your data is in a CSV file
rename_dict = {'Category_name' : 'category_name', 'Category' : 'category'}
df.rename(columns=rename_dict, inplace=True)
del rename_dict

df = df[['title', 'category_name']]  # Select relevant columns

# 2. Преобразование категорий в числовой формат
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['category_name'])

# 3. Разделение данных на обучающую и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['title'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# 4. Загрузка токенизатора BERT
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')


# Токенизация данных
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt", max_length=128)


train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)


# 5. Создание датасетов для обучения и тестирования
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


train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

# 6. Загрузка модели BERT для классификации
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(df['label'])))
model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased',
                                                      num_labels=len(set(df['label'])))

del df

# 7. Настройка параметров тренировки
training_args = TrainingArguments(
    output_dir='./results',  # Папка для сохранения моделей
    num_train_epochs=3,      # Количество эпох
    per_device_train_batch_size=8,  # Размер батча для обучения
    per_device_eval_batch_size=16,  # Размер батча для оценки
    warmup_steps=500,        # Линейный шаг разогрева
    weight_decay=0.01,       # Регуляризация
    logging_dir='./logs',     # Папка для логов
    logging_steps=10,
)

# 8. Создание объекта Trainer для обучения
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# 9. Обучение модели
trainer.train()

# 10. Оценка модели
metric = datasets.load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)


# Оценка на тестовых данных
results = trainer.evaluate()
print(results)


# 11. Предсказание категорий
def predict_category(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    predicted_category = label_encoder.inverse_transform(predictions.cpu().numpy())[0]
    return predicted_category


# Пример предсказания
print(predict_category("Samsung Galaxy S21"))
