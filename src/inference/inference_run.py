"""
Script loads the latest trained model, data for inference, predicts results and stores them.
"""
import os 
import sys
import warnings
import logging 
import joblib

import re
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

#Downloading nltk packages 
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

#Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting inference...\U0001f600")

#Configuring warnings
warnings.filterwarnings("ignore") 

#Defining ROOT directory, and appending it to the sys.path
# so that python know which file should be included within the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(ROOT_DIR))

#Defining data, models and results directories
from utils import get_root_dir
DATA_PATH = get_root_dir('data')
MODEL_PATH = get_root_dir('outputs/models')
RESULTS_PATH = get_root_dir('outputs/predictions')


def load_latest_model(model_path):
    '''Gets latest trained model'''
    logging.info("Loading latest model...")
    try:
        # Getting the list of models in the MODEL PATH
        models = [os.path.join(model_path, file) for file in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, file))]

        # Sorting models by their creation time
        sorted_models = sorted(models, key=os.path.getctime, reverse=True)

        #Checks for the avaliable models if there're no any raises Exception
        try:
            return sorted_models[0]
        except:
            logging.error(f"No pretrained model found... Make sure to train your model first!")
            sys.exit(1)

    except FileNotFoundError:
        logging.error(f"No pretrained model found... Make sure to train your model first!")
        sys.exit(1)

def load_and_preprocess_data(data_path: str):
    '''Loads inference data and returns it as DataFrame'''
    logging.info("Loading inference data...")
    try:
        logging.info("Preprocessing text data... (takes a little bit)")
        sentiment = pd.read_csv(os.path.join(data_path,"raw/test.csv"))
        #lower casing
        sentiment['review'] = sentiment['review'].str.lower() 

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

        #Tokenization
        def tokenize(text):
            return word_tokenize(text)
        sentiment['review_tokenized'] = sentiment['review'].apply(tokenize)

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

        #Converting back to string
        def convert_to_string(token):
            return " ".join(token)
        sentiment['cleaned_review'] = sentiment['cleaned_review'].apply(convert_to_string)

        STOPWORDS = set(stopwords.words('english'))
        def remove_stopwords(text):
            additional_words_to_keep = ['not', 'never', 'nor']
            return " ".join([word for word in str(text).split() if (word not in STOPWORDS) or (word in additional_words_to_keep)])
        sentiment["cleaned_review"] = sentiment["cleaned_review"].apply(lambda text: remove_stopwords(text))


        words_del = 'movie film one make would'.split()
        def remove_freq_words(review:str):
            return ' '.join([i for i in review.split() if i not in words_del])
        sentiment['cleaned_review'] = sentiment['cleaned_review'].apply(lambda text: remove_freq_words(text))

        sentiment['sentiment'] = sentiment['sentiment'].replace({'positive': 1, 'negative': 0})

        unigram_vec = TfidfVectorizer(max_features = 7000,)
        test_data = unigram_vec.fit_transform(sentiment['cleaned_review']).toarray()
            
        return test_data

    except Exception as ex:
        logging.error(f"An error occurred while loading inference data: {ex}")
        sys.exit(1)

def get_results(model:object, data: pd.DataFrame):
    '''Makes model to predict on inference data and gives results'''
    logging.info("Predicting results...")
    model = joblib.load(model)
    results = model.predict(data)
    return results

def save_results(result: pd.DataFrame,latest_model_name:str):
    '''Store the prediction results in 'outputs/predictions' directory with the name of the model predicted those results'''
    logging.info("Saving results...")
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    path = os.path.join(RESULTS_PATH, os.path.basename(latest_model_name)+"_results.csv")
    predictions = pd.DataFrame(result,columns=['sentiment'])
    predictions['sentiment'] = predictions['sentiment'].replace({1: 'positive', 0: 'negative'})
    predictions.to_csv(path, index=False)
    logging.info("Results saved successfully \N{grinning face}")


def main():
    '''Main Function'''
    latest_model = load_latest_model(MODEL_PATH)
    data = load_and_preprocess_data(DATA_PATH)
    results = get_results(latest_model,data)
    save_results(result=results, latest_model_name = latest_model)
   

if __name__ == "__main__":
    main()