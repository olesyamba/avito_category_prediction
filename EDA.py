# Importing necessary libraries
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import nltk
import re  # library for regular expression operations
import string  # for string operations

from nltk.corpus import stopwords  # module for stop words that come with NLTK
from nltk.stem import PorterStemmer  # module for stemming
from nltk.tokenize import TweetTokenizer  # module for tokenizing strings

# 1. Loading and Preprocessing the Dataset
df = pd.read_csv('data/train.csv')  # Assuming your data is in a CSV file
rename_dict = {'Category_name' : 'category_name', 'Category' : 'category'}
df.rename(columns=rename_dict, inplace=True)
del rename_dict

df = df[['title', 'category_name']]  # Select relevant columns


# Preprocess text (basic cleaning for now)
# def preprocess_text(text):
#     text = text.lower()
#     return text

nltk.download("stopwords")  # Set of stopwords
nltk.download('punkt')  # Split the text into number of sentences
nltk.download('wordnet')  # Lemmatize


# Create a function to clean text
def nltk_preprocess_text(text):
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)  # remove hyperlinks
    text = re.sub(r'#', '', text)  # Removing '#' hashtag
    text = re.sub(r'[^a-zA-Z]', " ", text)
    text = re.sub(r'@[A-Za-z0–9]+', '', text)  # Removing @mentions
    # Tokenize
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    text = tokenizer.tokenize(text)
    stopwords_rus = stopwords.words('russian')  # Import the russian stop words list from NLTK
    # Lemmatize
    lemmatize = nltk.WordNetLemmatizer()
    text = [lemmatize.lemmatize(word) for word in text if
            word not in stopwords_rus and word not in string.punctuation]
    # Instantiate stemming class
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    # Combine words
    text = " ".join(text)
    return text


df['title'] = df['title'].apply(nltk_preprocess_text)

# 2. Label Encoding the Categories
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

# 3. Train-Test Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['name'].tolist(),
    df['category_encoded'].tolist(),
    test_size=0.2,
    random_state=42
)

# 4. Tokenization using BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')


train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)


# 5. Creating Dataset for BERT
class CommodityDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


train_dataset = CommodityDataset(train_encodings, train_labels)
test_dataset = CommodityDataset(test_encodings, test_labels)

# 6. Load Pre-trained BERT for Sequence Classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=len(df['category_encoded'].unique()))

# 7. Training Arguments
training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,  # number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=10,
)

# 8. Trainer
trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset  # evaluation dataset
)

# 9. Fine-tune the Model
trainer.train()

# 10. Model Evaluation
results = trainer.evaluate()
print(f"Evaluation results: {results}")

# 11. Save the Fine-tuned Model
model.save_pretrained('fine_tuned_bert')
tokenizer.save_pretrained('fine_tuned_bert')


# Optional: Convert the category encoded labels back to their original form
def predict_category(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1)
    return label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]


# Test prediction
sample_name = "iPhone 12"
predicted_category = predict_category(sample_name)
print(f"Predicted Category for '{sample_name}': {predicted_category}")
