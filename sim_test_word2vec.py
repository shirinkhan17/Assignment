import numpy as np
import gensim
from scipy import spatial
from re import sub
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download('stopwords')
from gensim.models import KeyedVectors

#loading word2vec dataset from existing dataset.

#load word2vec datasets in a model using gensim
#model = KeyedVectors.load_word2vec_format('D:/test_A/GoogleNews-vectors-negative300.bin', binary=True)
#model = KeyedVectors.load_word2vec_format('D:/test_A/glove-wiki-gigaword-50.gz', binary=True)

W2V_PATH="D:/test_A/GoogleNews-vectors-negative300.bin.gz"
model = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
#map the vector into word in the datasets 
index2word_set = set(model.wv.index2word)
wordnet_lemmatizer = WordNetLemmatizer()

stop_words_l=stopwords.words('english')



def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess(doc):
    # Tokenize, clean up input document string
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [wordnet_lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if wordnet_lemmatizer.lemmatize(token) not in stop_words_l]
# Preprocess the documents, including the query string



with open("corpus.txt", "r", encoding="utf-8") as corpus:
        sentences = [sentence.replace("\n", "") for sentence in corpus.readlines()]
        corpus = [preprocess(document) for document in sentences]

print(corpus)

#calculate avaerage feature vector of words in a sentence  using word2vec dataset
def avg_feature_vector(words, model, num_features, index2word_set):
    
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        
        if word in index2word_set: #find the word in the mapped dataset model
            #print(model[word].shape)
            
            n_words += 1 #count the number of the word exist in the dataset
           
            feature_vec = np.add(feature_vec, model[word])  #include the feature of the word from dataset in a feature vector
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)  #calculate the feature of the sentence
    return feature_vec
k=0
for do in corpus: 
 s1_afv = avg_feature_vector(do, model=model, num_features=300, index2word_set=index2word_set)
 print("\n \n Word2Vec Similairity: \n", sentences[k], " \n") 
 i=0
 k=k+1
 for document in corpus: 
  s2_afv = avg_feature_vector(document, model=model, num_features=300, index2word_set=index2word_set)
  sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
  print(sentences[i], ": ", sim)
  i=i+1


#s1_afv = avg_feature_vector('I do like to eat banana ', model=model, num_features=300, index2word_set=index2word_set)
#s2_afv = avg_feature_vector('I do not like to eat banana', model=model, num_features=300, index2word_set=index2word_set)
#sim = 1 - spatial.distance.euclidean(s1_afv, s2_afv)