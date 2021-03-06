"""Main function"""
import logging
import argparse

from models.use_use import USECalculator
from models.use_elmo import ELMoCalculator
from models.use_bert import BERTCalculator
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
from re import sub
from gensim.utils import simple_preprocess
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()

stop_words_l=stopwords.words('english')
stop_words_l.append('like')
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
# From: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb

def preprocess(doc):
    # Tokenize, clean up input document string
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [wordnet_lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if wordnet_lemmatizer.lemmatize(token) not in stop_words_l]
# Preprocess the documents, including the query string



def main(config):
    models = {
        "use": USECalculator,
        "elmo": ELMoCalculator,
        "bert": BERTCalculator,
    }

    if config.model not in models:
        logging.error(f"The model you chosen is not supported yet.")
        return

    if config.verbose:
        logging.info(f"Loading the corpus...")

    with open("corpus.txt", "r", encoding="utf-8") as corpus:
        sentences = [sentence.replace("\n", "") for sentence in corpus.readlines()]
        sentences = [' '.join(preprocess(document)) for document in sentences]
        model = models[config.model](config, sentences)

        if config.verbose:
            logging.info(
                f'You chose the "{config.model.upper()}" as a model.\n'
                f'You chose the "{config.method.upper()}" as a method.'
            )

        model.calculate()

        if config.verbose:
            logging.info(f"Terminating the program...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="use", choices=["use", "elmo", "bert"]
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cosine",
        choices=[
            "cosine",
            "manhattan",
            "euclidean",
            "inner",
            "ts-ss",
            "angular",
            "pairwise",
            "pairwise-idf",
        ],
    )
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()
    main(args)
