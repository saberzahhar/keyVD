# -*- coding: utf-8 -*-
# Author: Saber Zahhar
# Date: August 2023

"""keyVD for keyword generation.

    this algorithm imports a vocabulary defined by the user 
    and computes Singular Value Decomposition to extract 
    from a text keywords that can be mapped to the vocabulary.

    example:

        # 0/ define variables
        text="this article is a generic example of a data scientist discussing pre procssing methods."
        vocabulary="pre-processing;data science;mathematics"
        parameters={"vector_size":300, "window":3, "min_count":1, "n_grams":3}
        n_keys=5
        sep=";"

        from keyVD import KeyVD

        # 1/ create a KeyVD keywords generator.
        generator = KeyVD()

        # 2/ load contents into the generator object.
        generator.load_vocabulary(vocabulary=vocabulary, 
                                  sep=sep)
        generator.load_text(text=text,
                            vector_size=parameters["vector_size"], 
                            window=parameters["window"], 
                            min_count=parameters["min_count"],
                            n_grams=parameters["n_grams"])

        # 3/ extract the n_keys-best cluster candidates and map them to our vocabulary.
        indexes = generator.keywords_generation(n_keys=n_keys)
        print(";".join(indexes))
"""
#%% Setup
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from scipy.spatial import distance
from operator import itemgetter
from numpy.linalg import svd
import numpy
import time

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import ngrams

try:
    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    remove_words = stopwords.words() + ['', ' ']
except:
    nltk.download('stopwords')
    nltk.download('punkt')
    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    remove_words = stopwords.words() + ['', ' ']

def get_ngrams(words, n=1):
    return [' '.join(phrases) for phrases in list(ngrams(words, n))]

class KeyVD():
    def __init__(self):
        self.vocab=None
        self.text=None
        self.vectors=None
        self.dicVecVocab=None
        self.dicVecText=None
        
    def load_vocabulary(self, vocabulary, sep = ';'):
        try:
            if '.txt' in vocabulary[-4:]:
                with open(vocabulary, 'r') as vocabulary_file:
                    self.vocab = set([' '.join(tokenizer.tokenize(word.lower())) 
                                      for line in vocabulary_file.readlines() 
                                      for word in line.split(sep) if word != "" and word != " "])
            else:
                self.vocab = set([' '.join(tokenizer.tokenize(word.lower())) 
                                  for line in vocabulary.split('\n') 
                                  for word in line.split(sep) if word != "" and word != " "])
            return self.vocab
        except:
            print('''Error loading vocabulary.\nPlease, make sure that you inserted either:\n\t- a path to a text file "/path/to/input.txt"\n\t- a string object''')
        
    def load_text(self, text, vector_size=300, window=3, min_count=1, n_grams=3):
        global tokenizer
        global remove_words
        try:
            if '.txt' in text[-4:]:
                with open(text, 'r') as text_file:
                    text = ' '.join(text_file.readlines()).lower()
                    self.text = []
                    for ngram in range(1, n_grams+1):
                        self.text.append([word for word in get_ngrams(tokenizer.tokenize(text), ngram) if word not in remove_words])
            else:
                text = text.lower()
                self.text = []
                for ngram in range(1, n_grams+1):
                    self.text.append([word for word in get_ngrams(tokenizer.tokenize(text), ngram) if word not in remove_words])                
            try:
                errors = list()
                if type(vector_size) in [float, int]:
                    if int(vector_size) != vector_size or vector_size <= 0:
                        errors.append('vector_size must be a non-negative integer.')
                else:
                    errors.append('vector_size must be a non-negative integer.')
                if type(min_count) in [float, int]:
                    if int(min_count) != min_count or min_count <= 0:
                        errors.append('min_count must be a non-negative integer.')
                else:
                    errors.append('min_count must be a non-negative integer.')
                if type(window) in [float, int]:
                    if int(window) != window or window <= 0:
                        errors.append('window must be a non-negative integer.')
                else:
                    errors.append('window must be a non-negative integer.')
                if len(errors) > 0:
                    raise Exception(f'''Error loading parameters:\n\t- ''' + "\n\t- ".join(errors))
                self.vectors = Word2Vec(sentences=self.text, min_count=min_count, vector_size=vector_size, window=window, seed=1).wv
            except Exception as e:
                print(e)

            self.dicVecVocab = {}
            for word in list(self.vocab):
                try:
                    self.dicVecVocab[word] = self.vectors[word]
                except:
                    subvec = numpy.zeros(vector_size)
                    for subword in tokenizer.tokenize(word):
                        try:
                            subvec += self.vectors[subword]
                        except:
                            continue
                    if subvec.sum() == 0:
                        errors.append(word)
                    else:
                        self.dicVecVocab[word] = subvec

            self.dicVecText = {}
            for word in self.vectors.key_to_index:
                try:
                    self.dicVecText[word] = self.vectors[word]
                except:
                    continue
            self.matText = numpy.array(list(self.dicVecText.values()))
        except:
            print('''Error loading text.\nPlease, make sure that you inserted either:\n\t- a path to a text file "/path/to/input.txt"\n\t- a string object''')

    def keywords_generation(self, n_keys=5, n_clusters=2):
        if self.vocab is None:
            print('You need to load a vocabulary first using .load_vocabulary()')
            return []
        elif self.text is None:
            print('You need to load a text first using .load_text()')
            return []
        elif self.dicVecText is None:
            print('You need to load a vocabulary and text first using .load_vocabulary() and .load_text()')
            return []        
        try:
            errors = list()
            if type(n_clusters) in [float, int]:
                if int(n_clusters) != n_clusters or n_clusters <= 0:
                    errors.append('n_clusters must be a non-negative integer.')
            else:
                errors.append('n_clusters must be a non-negative integer.')
            if len(errors) > 0:
                raise Exception(f'''Error loading parameters:\n\t- ''' + "\n\t- ".join(errors))
        except Exception as e:
            print(e)
        
        n_clusters = min(self.matText.shape[0], n_clusters*n_keys)
        textClusters = KMeans(n_clusters=n_clusters, 
                              init="k-means++", n_init=1, random_state=1).fit(self.matText).cluster_centers_
        clusterSVD, variancesSVD, embeddingSVD = numpy.linalg.svd(textClusters, full_matrices=True)
        n_axes = min(clusterSVD.shape[0], n_keys)

        keywords = list()
        for index_axis in range(n_axes):
            clusterCos = {index_cluster : (distance.cosine(clusterSVD[index_cluster,:], clusterSVD[:, index_axis]) - 1) 
                          for index_cluster in range(clusterSVD.shape[0])}
            candidate = sorted(clusterCos.items(), key=itemgetter(1), reverse=True)[0][0]
            vocabCos = {word : (distance.cosine(textClusters[candidate], self.dicVecVocab[word]) - 1) 
                        for word in self.dicVecVocab.keys()}
            try:
                keywords.append(sorted(vocabCos.items(), key=itemgetter(1), reverse=True)[0][0])
            except:
                continue
        keywords = set(keywords)
        return keywords