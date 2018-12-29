#!/usr/bin/python3
from gensim.models.word2vec import Word2Vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from itertools import islice
from gensim.models import KeyedVectors
from gensim.utils import SaveLoad
import timeit
import csv
#import cpickle as pickle
print("Loading pickle:")
bigram = SaveLoad.load("big_phrase.pickle")
trigram = SaveLoad.load("trig_phrase.pickle")
print("reading sentence from file")
with open("sentence.csv", "r") as f:
    reader = csv.reader(f)
    sentence_stream=list(reader)
##TODO word2vec
print('start word2vec') 
start = timeit.default_timer()
model = Word2Vec(trigram[bigram[sentence_stream]], size=100, window=5, min_count=40, workers=4)
print('finish word2vec time:{}'.format(timeit.default_timer() - start))
model.save("word2vec.pickle")
word_vectors = model.wv
KeyedVectors.save_word2vec_format(word_vectors, 'vectors.txt', binary=False)
word = KeyedVectors.load_word2vec_format('vectors.txt', binary=False)  # C text format
print(model.wv['computer'])
print(model.wv.most_similar(positive=['wall_st', 'u_s'], negative=['apple']))
print(model.wv.similarity('wall_st', 'wall_street'))
print(word['wall_st'])
