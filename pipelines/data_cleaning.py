import re
import string
import html
import emoji
import contractions
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from warnings import filterwarnings

filterwarnings('ignore')

# Download resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Lowercase
def to_lowercase(text):
    return text.lower()

# Expand Contractions
def expand_contractions(text):
    return contractions.fix(text)

# Remove HTML Tags
def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Remove Emojis
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

# Normalize Clothing Sizes
def normalize_clothing_sizes(text):
    # Convert height in feet/inches to cm (approximate)
    pattern = re.compile(r"(\d)'\s?(\d{1,2})\"?")
    def convert(match):
        feet = int(match.group(1))
        inches = int(match.group(2))
        cm = round((feet * 12 + inches) * 2.54)
        return f"{cm} cm"
    return pattern.sub(convert, text)

# Remove URLs
def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text)

# Remove Special Characters - Sentiment Analysis
def remove_special_chars(text):
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Keep alphanumeric only
    return text

# Remove Special Characters - Emotion Recognition
def remove_special_chars_emotion(text):
    # Keep: letters, digits, whitespace, and key emotional punctuation: ! ? . , : ; ' " - ...
    text = re.sub(r"[^A-Za-z0-9\s.,!?;:'\"-]", '', text)  
    return text

# Normalize Whitespace & Line Breaks
def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

# Lemmatization
# Initialize
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized)


# Data Cleaning Pipeline for Sentiment Analysis
def clean_review_sentiment(text):

    text = to_lowercase(text)
    text = expand_contractions(text)
    text = remove_html(text)
    text = remove_emojis(text)
    text = normalize_clothing_sizes(text)
    text = remove_urls(text)
    text = remove_special_chars(text)
    text = normalize_whitespace(text)
    text = lemmatize_text(text)

    return text


# Data Cleaning Pipeline for Emotion Recognition
def clean_review_emotion(text):
    
    text = expand_contractions(text)
    text = remove_html(text)
    text = normalize_clothing_sizes(text)
    text = remove_urls(text)
    text = remove_special_chars_emotion(text)
    text = normalize_whitespace(text)
    
    return text


# Data Cleaning Pipeline for ABSA
def clean_review_absa(text):
    
    text = to_lowercase(text)
    text = expand_contractions(text)
    text = remove_html(text)
    text = remove_emojis(text)
    text = normalize_clothing_sizes(text)
    text = remove_urls(text)
    text = remove_special_chars(text)
    text = normalize_whitespace(text)
    
    return text