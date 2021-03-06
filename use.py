# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 19:27:00 2021

@author: Shirin
"""


import spacy_universal_sentence_encoder
# load one of the models: ['en_use_md', 'en_use_lg', 'xx_use_md', 'xx_use_lg']
nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')
# get two documents
doc_1 = nlp('I try to do tasks as soon as possible and not leave them until last minute.')
doc_2 = nlp('I always make a list so I don\'t forget anything.')
# use the similarity method that is based on the vectors, on Doc, Span or Token
print(doc_1.similarity(doc_2[0:7]))