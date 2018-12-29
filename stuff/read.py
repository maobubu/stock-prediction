#!/usr/bin/python3
import timeit
import glob, os
from os import listdir
import string
import re
import json
import pandas as pd
from gensim.utils import SaveLoad
from gensim.models.word2vec import Word2Vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from itertools import islice
from gensim.models import KeyedVectors
import csv
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
#from nltk import pos_tag, word_tokenize
from nltk.tag import PerceptronTagger
from nltk.corpus import stopwords  # Import the stop word list

# Pywsd's Lemmatizer.
porter = PorterStemmer()
wnl = WordNetLemmatizer()
tagger = PerceptronTagger()
pos_tag = tagger.tag
tokenizer = RegexpTokenizer(r'\w+')


def lemmatize(ambiguous_word, pos=None, neverstem=True, 
              lemmatizer=wnl, stemmer=porter):
    """
    Tries to convert a surface word into lemma, and if lemmatize word is not in
    wordnet then try and convert surface word into its stem.
    This is to handle the case where users input a surface word as an ambiguous 
    word and the surface word is a not a lemma.
    """
    if pos:
        lemma = lemmatizer.lemmatize(ambiguous_word, pos=pos)
    else:
        lemma = lemmatizer.lemmatize(ambiguous_word)
    stem = stemmer.stem(ambiguous_word)
    # Ensure that ambiguous word is a lemma.
    if not wn.synsets(lemma):
        if neverstem:
            return ambiguous_word
        if not wn.synsets(stem):
            return ambiguous_word
        else:
            return stem
    else:
        return lemma

def penn2morphy(penntag, returnNone=False):
    morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
                  'VB':wn.VERB, 'RB':wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''

def word_tokenize(text,tokenize=tokenizer):
    #return tokenize.tokenize(text.lower())# doesn't remove stop words
    return [w for w in tokenize.tokenize(text.lower()) if not w in stopwords.words("english")]

def lemmatize_sentence(sentence, neverstem=False, keepWordPOS=False, 
                       tokenizer=word_tokenize, postagger=pos_tag, 
                       lemmatizer=wnl, stemmer=porter):
    words, lemmas, poss = [], [], []
    for word, pos in postagger(tokenizer(sentence)):
        pos = penn2morphy(pos)
        lemmas.append(lemmatize(word.lower(), pos, neverstem,
                                lemmatizer, stemmer))
        poss.append(pos)
        words.append(word)
    if keepWordPOS:
        return words, lemmas, [None if i == '' else i for i in poss]
    return lemmas


def new(sentence_stream):
    os.chdir('/home/huicheng/Documents/datas/reuters_new/reuters_new')
    for i, file in enumerate(glob.glob("*.json")):
        print(i)
        with open(file, "r") as data:
            D = pd.DataFrame(json.loads(line) for line in data)
        D = D.set_index('date')
        D.replace(['UPDATE\s\d-', "'s"], '', regex=True, inplace=True)
        D.replace(["\d+\S\d+",'\d+'], 'xxx', regex=True, inplace=True)  
        D['title'] = D['title'].str.replace('[{}]'.format(string.punctuation), ' ')
        D = D.drop_duplicates(subset=['title'], keep='first')
        sentence_stream += [list(filter(None,doc.strip().lower().split(" "))) for doc in D['title'].values]


# stop words issue

def extract(path, sentence_stream):
    count = 0
    subfolder_list = glob.glob(path +'/*')
    pbar = tqdm(total=len(subfolder_list))
    folders = [x[0] for x in os.walk(path)]
    for x in folders[1:-1]:
        #print(count)
        pbar.set_description('Extracting {}'.format(x))
        pbar.update(1)
        for i in listdir(x):
            count += 1
            with open(x + "/" + i, 'r') as f:
                lines = (line for line in islice(f, 7, None))
                try:
                    for line in lines:
                        line = re.sub("\d+\S\d+", 'xxx', line)#remove numbers
                        line = re.sub("\d+", 'xxx', line)#remove numbers
                        sentence_stream +=[lemmatize_sentence(i) for i in line.strip().split('. ')]
                        #sentence_stream +=[word_tokenize(i) for i in line.strip().split('. ')]
                        '''sentence_stream += [
                            list(filter(None, i.strip().lower().translate(
                                str.maketrans(string.punctuation, ' ' * len(string.punctuation))).split(' '))) for i
                            in line.strip().split('. ')]'''
                        # line=[i.strip().lower().replace('^[{}]'.format(string.punctuation), ' ') for i in
                        #                   line]
                except UnicodeDecodeError as e:
                    print(str(e))
    pbar.close()
        # get rid of the null in list.
        # replace punctuation with ' '
        # stop words


def main():
    sentence_stream = []
    start = timeit.default_timer()
    print('start the reuters')
    extract("/home/huicheng/Documents/datas/ReutersNews106521", sentence_stream)
    print(len(sentence_stream))
    print('start the bloombergs')
    #extract("/home/huicheng/Documents/datas/20061020_20131126_bloomberg_news", sentence_stream)
    print(len(sentence_stream))
    print('start ours')
    #new(sentence_stream)
    print('before:{}'.format(len(sentence_stream)))
    print(timeit.default_timer() - start)
    start = timeit.default_timer()
    sentence_stream = list(filter(None, sentence_stream))
    print('after:{}'.format(len(sentence_stream)))
    print(timeit.default_timer() - start)
    print('generating phrase and word2vec')
    start = timeit.default_timer()
    os.chdir("/home/huicheng/Documents/datas/")
    with open("sentence.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(sentence_stream)
    phrases = Phrases(sentence_stream,min_count=500, threshold=2)
    bigram = Phraser(phrases)
    # print(list(bigram[sentence_stream]))
    print(bigram['u', 's', 'wall', 'st', 'wall', 'street','s','p','500','s','p','xxx'])
    ##TODO phrase
    bigram.save("big_phrase.pickle")
    print('finish phrase time:{}'.format(timeit.default_timer() - start))
    print('start trigram')
    start = timeit.default_timer()
    phrases = Phrases(bigram[sentence_stream],min_count=500, threshold=2)
    trigram = Phraser(phrases)
    trigram.save("trig_phrase.pickle")
    print(trigram[bigram['u', 's', 'wall', 'st', 'wall', 'street','bank','of','america','s','p','500','s','p','xxx']])
    print('finish phrase time:{}'.format(timeit.default_timer() - start))
    '''##TODO word2vec
    print('start word2vec')
    start = timeit.default_timer()
    model = Word2Vec(bigram[sentence_stream], size=300, window=5, min_count=5, workers=4)
    print('finish word2vec time:{}'.format(timeit.default_timer() - start))
    model.save("word2vec.pickle")
    print(model.wv['computer'])
    print(model.wv.most_similar(positive=['wall_st', 'u_s'], negative=['apple']))
    print(model.wv.similarity('wall_st', 'wall_street'))
    word_vectors = model.wv
    KeyedVectors.save_word2vec_format(word_vectors, 'vectors.txt', binary=False)
    word = KeyedVectors.load_word2vec_format('vectors.txt', binary=False)  # C text format
    print(word['wall_st'])'''


if __name__ == '__main__':
    main()
