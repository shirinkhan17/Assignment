import logging

import tensorflow as tf
import tensorflow_hub as hub

from utils.basic import *
from utils.ts_ss import triangle_sector_similarity

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"


class USECalculator:
    def __init__(self, config, sentences):
        self.sentences = sentences
        self.method = config.method
        self.verbose = config.verbose

    def calculate(self):
        methods = {
            "cosine": cosine_sim,
            "manhattan": manhattan_dist,
            "euclidean": euclidean_dist,
            "angular": angular_distance,
            "inner": inner_product,
            "ts-ss": triangle_sector_similarity,
        }

        if self.method not in methods:
            logging.error(f"The method you chosen is not supported yet.")
            return False

        model = hub.load(module_url)
        if self.verbose:
            logging.info(f"Now embedding sentence...")

        embeddings = model(self.sentences)
        method = methods[self.method]

        if self.verbose:
            logging.info("USE Calculating similarity between sentences...")

        similarity = method(embeddings, embeddings)
        i=0
        for s in similarity:
         print("\n \n USE Similarity: \n", self.sentences[i], "\n")
         i=i+1
         k=0
         for sim in s:
          print(self.sentences[k],":",sim)
          k=k+1
        
        #plot_similarity(self.sentences, similarity, self.method)
