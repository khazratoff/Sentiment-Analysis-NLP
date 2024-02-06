"""
Script loads and prepares the data, runs the training, and saves the model.
"""
# Imports
import os
import sys
import logging
import warnings
import re
import joblib

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score 
from sklearn.linear_model import LogisticRegression     
from sklearn.model_selection import train_test_split    

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

#Downloading nltk packages 
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

#Configuring warnings
warnings.filterwarnings("ignore") 

#Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting...\U0001f600")


#Defining ROOT directory, and appending it to the sys.path
# so that python know which file should be included within the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(ROOT_DIR))

#Defining data and model directories
from utils import get_root_dir
DATA_PATH = get_root_dir('data')
MODEL_PATH = get_root_dir('outputs/models')

#Loading dataset
logging.info(f"Loading 'Polar Movie Reviews' dataset...")
sentiment = pd.read_csv(os.path.join(DATA_PATH,'raw/train.csv'))


#Preprocessing data
logging.info(f"Step 1 | Preprocessing data\n=======================================================================\n")

logging.info(f"1.Lower casing...")
#lower casing
sentiment['review'] = sentiment['review'].str.lower() 

logging.info(f"2.Removing unnecessary characters (HTML tags, URLs, numbers, etc)..." )
#removing urls
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
sentiment["review"] = sentiment["review"].apply(lambda text: remove_urls(text))

#removing html tags (they appear so often in text)
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)
sentiment["review"] = sentiment["review"].apply(lambda text: remove_html(text))    

#removing non-alpha-numeric chars
def remove_non_alpha_numeric(text):
    return re.sub('[^a-zA-Z]',' ',text)
sentiment["review"] = sentiment["review"].apply(lambda text: remove_non_alpha_numeric(text))  

logging.info(f"3.Tokenizing...")
#Tokenization
def tokenize(text):
    return word_tokenize(text)
sentiment['review_tokenized'] = sentiment['review'].apply(tokenize)

logging.info(f"4.Lemmatizing... (takes a little bit)")
# Lemmatization
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

sentiment["lemma_review"] = sentiment["review"].apply(lambda text: lemmatize_words(text))
sentiment['lemma_review_tokenized'] = sentiment['lemma_review'].apply(tokenize)

#Clearing small length words
def clean_small_length(token):
    return [i for i in token if len(i)>2]
sentiment['cleaned_review'] = sentiment['lemma_review_tokenized'].apply(clean_small_length)

logging.info(f"Converting back to string...")
#Converting back to string
def convert_to_string(token):
    return " ".join(token)
sentiment['cleaned_review'] = sentiment['cleaned_review'].apply(convert_to_string)

logging.info(f"Filtering and removing stopwords, frequent and small words...")
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    additional_words_to_keep = ['not', 'never', 'nor']
    return " ".join([word for word in str(text).split() if (word not in STOPWORDS) or (word in additional_words_to_keep)])
sentiment["cleaned_review"] = sentiment["cleaned_review"].apply(lambda text: remove_stopwords(text))


words_del = 'movie film one make would'.split()
def remove_freq_words(review:str):
    return ' '.join([i for i in review.split() if i not in words_del])
sentiment['cleaned_review'] = sentiment['cleaned_review'].apply(lambda text: remove_freq_words(text))

#Saving cleaned data to "data/processed dir"
logging.info(f'Saving data to "data/processed" path...')
def save_clean_data():
    '''Saves cleaned data to 'data/processed' directory'''
    clean_data_path = os.path.join(DATA_PATH,'processed')
    if not os.path.exists(clean_data_path):
        os.makedirs(clean_data_path)
        clean_train_data_path = os.path.join(clean_data_path,'clean_train.csv')
        sentiment.to_csv(clean_train_data_path,index=False)
        logging.info(f'Cleaned train data saved successfully!\n===============================================================\n')


def train_model():
    '''Perfoms word embedding, trains the model and evaluates it'''
    logging.info(f'Step 2 | Training model\n=================================================================\n')
    logging.info(f'Encoding labels...')
    sentiment['sentiment'] = sentiment['sentiment'].replace({'positive': 1, 'negative': 0})

    logging.info(f'Splitting data into train and test parts...')
    X_train, X_test, y_train, y_test = train_test_split(sentiment['cleaned_review'],sentiment['sentiment'],test_size=0.1,shuffle=True)

    logging.info(f'Word embedding...')
    unigram_vec = TfidfVectorizer(max_features = 7000,)
    X_train_unigram = unigram_vec.fit_transform(X_train).toarray() # fit and transform train
    X_test_unigram = unigram_vec.transform(X_test).toarray() # transform test

    logging.info(f'Model training...')
    model = LogisticRegression(penalty='l2')
    model.fit(X_train_unigram, y_train)
    predictions = model.predict(X_test_unigram)
    log_reg_acc = accuracy_score(y_test, predictions)
    logging.info(f"Model Accuracy: {log_reg_acc * 100:.2f}%")
    logging.info(f"Precision: {precision_score(y_test,predictions)* 100:.2f}%")
    logging.info(f"Recall: {recall_score(y_test,predictions)* 100:.2f}%")
    logging.info(f"F1 score: {f1_score(y_test,predictions)* 100:.2f}%")
    return model



def save_trained_model(model):
    '''Saves the model in 'output/models' directory'''

    #Saving the model to the models folder
    logging.info("Saving the model...")
    model_name = os.environ.get("MODEL_NAME", "final_model.joblib")
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    model_path = os.path.join(MODEL_PATH,model_name)
    joblib.dump(model,model_path)
    logging.info(f'"{model_name}" saved successfully! \N{grinning face}')



def main():
    '''Main Function'''
    save_clean_data()
    model = train_model()
    save_trained_model(model)


if __name__ == "__main__":
    main()