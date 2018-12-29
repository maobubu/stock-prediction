import json
import pandas as pd
import pickle as pkl
import re, os, glob
import numpy as np
import re
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
tokenizer = RegexpTokenizer(r'\.|\w+')


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
    return tokenize.tokenize(text.lower())  # doesn't remove stopwords
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
    types = ['title', 'abstract', 'article']
    start = timeit.default_timer()
    print('start processing Ding_abstract data')
    article = defaultdict(list)
    with open('preporcessed_punkt_reuters_bloom(f2)_train.json',
              "r") as data:  # title+ abstract + article
        title = pd.DataFrame(json.loads(line, strict=False) for line in data).set_index('date')
    print('original shape', title.shape)
    title['title'] = title['title'].apply(lambda x: [x])
    #title = title.applymap(lambda x: np.nan if not x else x)
    ##TODO drop nan
    title = title.dropna(axis=0, how='any')  # drop the row that's incomplete
    title = title.sort_index()
    print('after first drop nan', title.shape)
    title = title.applymap(lambda x: '.'.join(x))
    title = title.replace(['UPDATE\s\d-', "'s"], '', regex=True)  # becareful of using inplace
    title = title.replace(['wo not'], 'would not', regex=True)  # becareful of using inplace
    title = title.replace(['bln'], 'billion', regex=True)  # becareful of using inplace
    title = title.replace(['mln'], 'million', regex=True)  # becareful of using inplace
    title = title.replace(['pct'], 'percent', regex=True)  # becareful of using inplace
    title = title.replace(['j j'], 'j_j', regex=True)  # becareful of using inplace
    title = title.replace(['h r'], 'h_r', regex=True)  # becareful of using inplace
    title = title.replace(['s korean'], 'korean', regex=True)  # becareful of using inplace
    title = title.replace(['\d+\S\d+', '\d+', 'xxx xxx', 'xxx xxx xxx'], 'xxx',
                          regex=True)  # becareful of using inplace
    ##TODO only work on title
    print('finish dealing with manual replace')
    title['titles']=title['title'].apply(convert, args=(bigram, trigram, False))
    '''
    for i in types:
        # title[i] = title[i].str.replace('[{}]'.format(string.punctuation), ' ')
        title[i] = title[i].apply(convert, args=(bigram, trigram, False))
        #title = title.drop_duplicates(subset=[i], keep='first')
        print('after dropping ' + i + ' is {}'.format(title.shape))
    '''
    #title['title'] = title['title'].apply(lambda x: x if len(x) >= 20 else np.nan)
    title['abstract'] = title['abstract'].apply(lambda x: x if len(x) >= 100 else np.nan)
    title['article'] = title['article'].apply(lambda x: x if len(x) >= 100 else np.nan)
    ##TODO drop nan
    #title = title.dropna(axis=0, how='any')  # drop the row that's incomplete
    print('after second drop nan', title.shape)
    #title = title.applymap(lambda x: x.split('.'))
    ##TODO the news start from 2006-10-20 to 2013-11-20 reuters, bloomberg to 2013-11-26
    train = title['2006-10-20':'2012-07-21']  # from the beginning 2006-10-20 plus 30 days after, just in case
    validate = title['2012-05-22':'2013-04-01']  # plus 30 days before and after, just in case
    test = title['2013-02-12':'2013-11-20']  # plus 30 days before and after, just in case
    stop = timeit.default_timer()
    print("run time for ding:", stop - start)
    os.chdir(path)
    # os.chdir('/home/jialong/Documents/phrase_embedding/yunke_' + types + '/')
    train.reset_index().to_csv("train.csv", index=False, encoding='utf-8')
    validate.reset_index().to_csv("validate.csv", index=False, encoding='utf-8')
    test.reset_index().to_csv("test.csv", index=False, encoding='utf-8')
    for i in types:
        build_dictionary(title, path, i)


def build_dictionary(filepaths, dst_path, vocab):
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


def convert(reviews, bigram, trigram, remove_stopwords=False):
    # letters_only = re.sub("[^a-zA-Z0-9]", " ", str(reviews))
    # words = list(filter(None, letters_only.lower().split()))
    words = word_tokenize(reviews)  # tokenize and remove punctuation
    if remove_stopwords:  # remove stopwords
        words = [w for w in words if not w in stopwords.words("english")]
    words = lemmatize_sentence(words)  # lemma
    words = trigram[bigram[words]]  # to phrase
    return " ".join(words)  # + ' .'


def main():
    # nltk.download('stopwords')
    print("start pre-processing the data")
    bigram = SaveLoad.load("big_phrase.pickle")
    trigram = SaveLoad.load("trig_phrase.pickle")
    label_one = pd.read_pickle("label_one.pickle")['2006-11-20':'2013-11-21']  # ['2006-11-20':'2013-11-21']
    path = 'ding_new_11/'
    length = label_one.shape[0]
    train = label_one[0:int(length * 0.8)]
    validate = label_one[int(length * 0.8):int(length * 0.9)]
    test = label_one[int(length * 0.9):-1]
    train.reset_index().to_csv(path + "train_label.csv", index=False, encoding='utf-8')
    validate.reset_index().to_csv(path + "validate_label.csv", index=False,
                                  encoding='utf-8')
    test.reset_index().to_csv(path + "test_label.csv", index=False, encoding='utf-8')
    print("starting the training selecting phase")
    Ding_abstract(label_one, bigram, trigram, path)


if __name__ == '__main__':
    main()
