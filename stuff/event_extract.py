import json
import pandas as pd
import pickle as pkl
import re, os, glob
import numpy as np
import sys
from collections import defaultdict
import nltk
import string
from gensim.models import Phrases
from gensim.utils import SaveLoad
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords  # Import the stop word list
import timeit
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk import pos_tag, word_tokenize
from nltk.tag import PerceptronTagger
from collections import OrderedDict

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
    morphy_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ,
                  'VB': wn.VERB, 'RB': wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''


def word_tokenize(text, tokenize=tokenizer):
    return tokenize.tokenize(str(text).lower())  # doesn't remove stopwords
    # return [w for w in tokenize.tokenize(text.lower()) if not w in stopwords.words("english")]


def lemmatize_sentence(sentence, neverstem=False, keepWordPOS=False,
                       tokenizer=word_tokenize, postagger=pos_tag,
                       lemmatizer=wnl, stemmer=porter):
    words, lemmas, poss = [], [], []
    for word, pos in postagger(sentence):  # change tokenizer(sentence) to sentence
        pos = penn2morphy(pos)
        lemmas.append(lemmatize(word.lower(), pos, neverstem,
                                lemmatizer, stemmer))
        poss.append(pos)
        words.append(word)
    if keepWordPOS:
        return words, lemmas, [None if i == '' else i for i in poss]
    return lemmas


def Ding_abstract(label, bigram, trigram, path='/Users/maobu/Dropbox/stock/data/ding/'):
    #type = ['e1', 'relation', 'e2']
    start = timeit.default_timer()
    print('start processing event data')
    article = defaultdict(list)
    bloomberg = pd.read_table('event/bloomberg_event.txt', names=['date', 'e1', 'relation', 'e2']).set_index('date')
    reuters = pd.read_table('event/reuters_event.txt', names=['date', 'e1', 'relation', 'e2']).set_index('date')
    title = reuters.append(bloomberg)
    title = title.sort_index()
    title = title.replace(['UPDATE\s\d-', "'s"], '', regex=True)  # becareful of using inplace
    title = title.replace(['\d+\S\d+', '\d+', 'xxx xxx', 'xxx xxx xxx'], 'xxx',
                          regex=True)  # becareful of using inplace
    print(title.shape)
    title['e2'] = title['e2'].apply(convert, args=(bigram, trigram, True))
    title = title.applymap(lambda x: np.nan if isinstance(x, str) and (not x or x.isspace()) else x)
    title = title.dropna(axis=0, how='any')  # drop the row that's incomplete
    print(title.shape)
    title['combine'] = title.apply(
        lambda x: ' '.join(x['e1'].split('_') + x['relation'].split('_') + x['e2'].split('_')), axis=1)
    # title = title.str.replace('[{}]'.format(string.punctuation), ' ')
    print('after dropping is {}'.format(title.shape))
    title['combine'] = title['combine'].apply(convert, args=(bigram, trigram, False))
    ##TODO the news start from 2006-10-20 to 2013-11-20 reuters, bloomberg to 2013-11-26
    train = title['2006-10-20':'2012-07-21']  # from the beginning 2006-10-20 plus 30 days after, just in case
    validate = title['2012-05-22':'2013-04-01']  # plus 30 days before and after, just in case
    test = title['2013-02-12':'2013-11-27']  # plus 30 days before and after, just in case
    stop = timeit.default_timer()
    print("run time for ding:", stop - start)
    os.chdir(path)
    # os.chdir('/home/jialong/Documents/phrase_embedding/yunke_' + types + '/')
    train.reset_index().to_csv("train.csv", index=False, encoding='utf-8')
    validate.reset_index().to_csv("validate.csv", index=False, encoding='utf-8')
    test.reset_index().to_csv("test.csv", index=False, encoding='utf-8')
    build_dictionary(title, path)


def build_dictionary(filepaths, dst_path, vocab='combine'):
    word_freqs = OrderedDict()
    print('processing the vocab cases for ' + vocab)
    for i in filepaths[vocab].values:
        words_in = i.strip().split(' ')
        for w in words_in:
            if w not in word_freqs:
                word_freqs[w] = 0
            word_freqs[w] += 1

    words = list(word_freqs.keys())
    freqs = list(word_freqs.values())

    sorted_idx = np.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    worddict = OrderedDict()
    worddict['_PAD_'] = 0  # default, padding
    worddict['_UNK_'] = 1  # out-of-vocabulary
    worddict['_BOS_'] = 2  # begin of sentence token
    worddict['_EOS_'] = 3  # end of sentence token

    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii + 4

    with open('vocab_cased_' + vocab + '.pickle', 'wb') as f:
        pkl.dump(worddict, f)

    print('Dict size', len(worddict))
    print('Done')


def convert(reviews, bigram, trigram, remove_stopwords=True):
    # letters_only = re.sub("[^a-zA-Z0-9]", " ", str(reviews))
    # words = list(filter(None, letters_only.lower().split()))
    words = word_tokenize(reviews)  # tokenize and remove punctuation
    if remove_stopwords:  # remove stopwords
        words = [w for w in words if not w in stopwords.words("english")]
    words = trigram[bigram[words]]  # to phrase
    words = lemmatize_sentence(words)  # lemma
    return " ".join(words)  # + ' .'


def main():
    # nltk.download('stopwords')
    print("start pre-processing the data")
    bigram = SaveLoad.load("big_phrase.pickle")
    trigram = SaveLoad.load("trig_phrase.pickle")
    label_one = pd.read_pickle("label_one_new.pickle")['2006-11-20':'2013-11-27']  # ['2006-11-20':'2013-11-21']
    path = '/Users/maobu/Dropbox/stock/data/ding/'
    length = label_one.shape[0]
    train = label_one[0:int(length * 0.8)]
    validate = label_one[int(length * 0.8):int(length * 0.9)]
    test = label_one[int(length * 0.9):-1]
    train.reset_index().to_csv(path + "train_label_new.csv", index=False, encoding='utf-8')
    validate.reset_index().to_csv(path + "validate_label_new.csv", index=False,
                                  encoding='utf-8')
    test.reset_index().to_csv(path + "test_label_new.csv", index=False, encoding='utf-8')
    print("starting the training selecting phase")
    #Ding_abstract(label_one, bigram, trigram, path)


if __name__ == '__main__':
    main()
