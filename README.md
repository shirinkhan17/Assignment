# Sentence Similarity Calculator
This repo contains various ways to calculate the similarity between source and target sentences. You can choose **the pre-trained models** you want to use such as _ELMo_, _BERT_ and _Universal Sentence Encoder (USE)_.

And you can also choose **the method** to be used to get the similarity:

    1. Cosine similarity
    2. Manhattan distance
    3. Euclidean distance
    4. Angular distance
    5. Inner product
    6. TS-SS score
    7. Pairwise-cosine similarity
    8. Pairwise-cosine similarity + IDF
    
You can experiment with (**The number of models**) x (**The number of methods**) combinations!
    
<br/>

## Installation

- This project is developed under **conda** enviroment
- After cloning this repository, you can simply install all the dependent libraries described in `requirements.txt` with `bash install.sh`

```
conda create -n sensim python=3.7
conda activate sensim
git clone https://github.com/Huffon/sentence-similarity.git
cd sentence-similarity
bash install.sh
```

<br/>

## Usage
- To **test** your own sentences, you should fill out [**corpus.txt**](corpus.txt) with sentences as below:

```
I ate an apple.
I went to the Apple.
I ate an orange.
...
```

- Then, **choose** the **model** and **method** to be used to calculate the similarity between source and target sentences

```
python sensim.py
    --model    MODEL_NAME  [use, bert, elmo]
    --method   METHOD_NAME [cosine, manhattan, euclidean, inner,
                            ts-ss, angular, pairwise, pairwise-idf]
    --verbose  LOG_OPTION (bool)
```

<br/>

## Examples
- In this section, you can see the example result of `sentence-similarity`
- As you know, there is a no **silver-bullet** which can calculate **_perfect similarity_** between sentences
- You should conduct various experiments with your dataset
    - _**Caution**_: `TS-SS score` might not fit with **sentence** similarity task, since this method originally devised to calculate the similarity between long documents
- **Result**:

Similairty:
1. I like walking when I travel	

I like to eat when I travel: 0.69121134 (S)
I feel lonely in life: 0.12975985 (DIS)
flying kite: 0.44844344 (S)
Playing football: 0.38583282 (S)
I do cycle and biking: 0.38583282 (S) 
Rock climbing: 0.35936177 (S)
I find it very difficult to get up in the morning: 0.14349224 (DIS)
I spent much time in internet: 0.09431196 (DIS)


<br/>
2. Trekking in the primary forest

I feel lonely in life: 0.09159722 (DIS)
Playing football: 0.087622076 (DIS)
I do cycle and biking: 0.3741088 (S)
I love camping in the mountain: 0.35939074 (S)
I find it very difficult to get up in the morning: 0.08324303 (DIS)
I spent much time in internet: 0.036347397 (DIS)
I sleep when I stay home: 0.04717419 (DIS)
Listening to music: 0.014895751 (DIS)
I have two mobiles: 0.03893737 (DIS)
<br/>
3. Watching bird 

I like to eat when I travel: 0. 33489358 (S)
I like walking when I travel: 0.36949763 (S)
I feel lonely in life: 0.09789418 (DIS)
flying kite: 0.40492806 (DIS)
I like swimming in the sea: 0.3177985 (S)
Playing football: 0.41372964 (S)
I spent much time in internet: 0.09161192 (DIS)
Listening to music: 0.38825262 (S)
<br/>
4. I feel lonely in life
I like walking when I travel: 0.12975985 (S)
Trekking in the primary forest: 0.09159722 (DIS)
watching bird: 0.09789418 (DIS)
I like swimming in the sea: 0.048059747 (DIS)
I find it very difficult to get up in the morning: 0.13500313 (DIS)
I spent much time in internet: 0.23604393 (DIS)
I sleep when I stay home: 0.40517464 (S)
<br/>
5. I like swimming in the sea

Flying kite: 0.39516008 (S)
Playing football: 0.41546327 (S)
I do cycle and biking: 0.30392492 (S)
I find it very difficult to get up in the morning: 0.13998166 (DIS)
I spent much time in internet: 0.05746212 (DIS)
I sleep when I stay home: 0.1609051 (DIS)
Listening to music: 0.14645165 (DIS)

<br/>

## References
### Papers
- [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)
- [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
- [A Hybrid Geometric Approach for Measuring Similarity Level Among Documents and Document Clustering](https://ieeexplore.ieee.org/document/7474366/metrics#metrics)

<br/>

### Libraries
- [TF-hub's Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/2)
- [Allen NLP's ELMo](https://github.com/allenai/allennlp)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [BERTScore](https://github.com/Tiiiger/bert_score)
- [Vector Similarity](https://github.com/taki0112/Vector_Similarity)

<br/>

### Articles
