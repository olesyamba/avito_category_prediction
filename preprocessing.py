import re
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import contractions
import unicodedata
from spellchecker import SpellChecker
import nlpaug.augmenter.word as naw
from sklearn.feature_extraction.text import CountVectorizer
import pymorphy2  # For better Russian lemmatization

# Load spacy model and stopwords for Russian
nlp = spacy.load("ru_core_news_sm")
stop_words = set(stopwords.words('russian'))

# Initialize spell checker and augmenters for Russian
spell = SpellChecker(language='ru')  # Use Russian language for spell correction
morph = pymorphy2.MorphAnalyzer()  # Russian lemmatizer using pymorphy2
synonym_aug = naw.SynonymAug(
    aug_src='wordnet')  # Synonym-based augmentation, not Russian-specific but works for simple tasks


# Function to apply spell check correction
def correct_spelling(tokens):
    corrected_tokens = [spell.correction(token) if spell.unknown([token]) else token for token in tokens]
    return corrected_tokens


# Function for generating n-grams
def generate_ngrams(text, n=2):
    vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='word',
                                 token_pattern=r'(?u)\b\w+\b')  # Word boundaries for Russian
    ngrams = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()


# Russian Lemmatization
def lemmatize_russian(tokens):
    return [morph.parse(token)[0].normal_form for token in tokens]  # Use pymorphy2 for lemmatization


# Enhanced Text Preprocessing Function for Russian Dataset
def advanced_preprocess_rus(text, remove_stopwords=True, lemmatize=True, expand_contractions=False,
                            remove_urls=True, preserve_entities=True, correct_spell=True, augment_text=True,
                            add_ngrams=True, ngram_range=2):
    # 1. Normalize Unicode Characters
    text = unicodedata.normalize('NFKD', text)

    # 2. Expand Contractions (e.g., "нельзя" -> "не можно") - Not very common in Russian, so kept as False by default
    if expand_contractions:
        text = contractions.fix(text)

    # 3. Remove URLs
    if remove_urls:
        text = re.sub(r'http\S+|www.\S+', '', text)

    # 4. Remove Emails
    text = re.sub(r'\S*@\S*\s?', '', text)

    # 5. Remove Special Characters & Digits, retain words
    text = re.sub(r'[^а-яА-ЯёЁ\s]', '', text)

    # 6. Lowercase the text
    text = text.lower()

    # 7. Remove Repeated Characters (e.g., "уууу" -> "у")
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # 8. Tokenize the Text (using NLTK or any other tokenization library that works well with Russian)
    tokens = word_tokenize(text)

    # 9. Remove Stopwords (Optional)
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]

    # 10. Spell Correction (Optional)
    if correct_spelling:
        tokens = correct_spelling(tokens)

    # 11. Lemmatization (Optional)
    if lemmatize:
        tokens = lemmatize_russian(tokens)

    # 12. Preserve Named Entities (Optional, spacy will help recognize entities like locations, people, organizations)
    if preserve_entities:
        doc = nlp(text)
        entity_tokens = [ent.text for ent in doc.ents]
        tokens = entity_tokens + tokens

    # 13. Text Augmentation (Optional)
    if augment_text:
        tokens = synonym_aug.augment(' '.join(tokens))
        tokens = tokens.split()  # Re-tokenize the augmented text

    # 14. Re-join tokens back to string format
    preprocessed_text = ' '.join(tokens)

    # 15. Generate N-Grams (Optional)
    if add_ngrams:
        ngrams = generate_ngrams(preprocessed_text, ngram_range)
        preprocessed_text += ' ' + ' '.join(ngrams)

    return preprocessed_text


# Example Usage with Russian Text
sample_text_ru = """
рено меган 3
"""

preprocessed_text_ru = advanced_preprocess_rus(
    sample_text_ru, correct_spell=True, augment_text=False, add_ngrams=False, ngram_range=3
)

print(f"Original Text: {sample_text_ru}")
print(f"Preprocessed Text: {preprocessed_text_ru}")
