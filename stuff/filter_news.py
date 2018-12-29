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


def fil_sub(reviews, fill=None):
    words = word_tokenize(str(reviews))  # tokenize and remove punctuation
    if any(x in words for x in fill):
        return reviews
    return np.nan


def filter_news(title, bigram, trigram):
    fill = set()
    # sp = pd.read_csv('sp500.csv', names=['Symbol', 'Name', 'market', 'number'])
    sp = pd.read_csv('sp100.csv')
    sp = sp.dropna(axis=0, how='any')  # drop the row that's incomplete
    sp['Symbol'] = sp['Symbol'].str.lower()
    ##TODO the following steps make sure that all the symbols and company names are included,despite format
    # TODO Do the phrase
    sp['phrase'] = sp['Name'].apply(convert, args=(bigram, trigram, False, True, False)).replace([' \. ', ' \.'], ' ',
                                                                                                 regex=True)
    sp['phrase'] = sp['phrase'].replace(['\\bu s\\b', '\\bs p\\b', '\\bat t\\b', '\\b\d+\\b'],
                                        ['u_s', 's_p', 'at_t', 'xxx'],
                                        regex=True)  # becareful of using inplace
    sp['phrase'] = sp['phrase'].replace(
        ['class_\w', 'class \w', ' inc\\b', ' corp\\b', ' company\\b', ' com\\b', ' group\\b', ' store\\b',
         ' co\\b',
         ' plc\\b', "\\b\w\\b"],
        '',
        regex=True)
    # TODO not do the phrase
    sp['no_phrase'] = sp['Name'].apply(convert, args=(bigram, trigram, False, False, False)).replace([' \. ', ' \.'],
                                                                                                     ' ',
                                                                                                     regex=True)
    sp['no_phrase'] = sp['no_phrase'].replace(['\\bu s\\b', '\\bs p\\b', '\\bat t\\b', '\\b\d+\\b'],
                                              ['u_s', 's_p', 'at_t', 'xxx'],
                                              regex=True)  # becareful of using inplace
    sp['no_phrase'] = sp['no_phrase'].replace(
        ['class_\w', 'class \w', ' inc\\b', ' corp\\b', ' company\\b', ' com\\b', ' group\\b', ' store\\b',
         ' co\\b',
         ' plc\\b', "\\b\w\\b"],
        '',
        regex=True)
    ##TODO connet all together
    sp['connect'] = sp['Name'].apply(convert, args=(bigram, trigram, False, False, False)).replace(
        ['\\bu s\\b', '\\bs p\\b', '\\bat t\\b', '\\b\d+\\b'],
        ['u_s', 's_p', 'at_t', 'xxx'],
        regex=True).replace(
        [' \. ', ' \.', "\\b\w\\b", 'class'], ' ',
        regex=True).apply(
        lambda x: '_'.join(x.split()))
    ##TODO remove last word
    sp['rm_last'] = sp['Name'].apply(convert, args=(bigram, trigram, False, False, False)).replace(
        ['\\bu s\\b', '\\bs p\\b', '\\bat t\\b', '\\b\d+\\b'],
        ['u_s', 's_p', 'at_t', 'xxx'],
        regex=True).replace(
        [' \. ', ' \.', "\\b\w\\b"], ' ',
        regex=True).apply(lambda x: ' '.join(x.split()[:-1]))
    sp['rm_last_connect'] = sp['rm_last'].apply(lambda x: '_'.join(x.split()))
    sp['phrase_connect'] = sp['phrase'].apply(lambda x: '_'.join(x.split()))
    sp['no_phrase_connect'] = sp['no_phrase'].apply(lambda x: '_'.join(x.split()))
    fill.update(sp['no_phrase'].values.tolist())
    fill.update(sp['phrase'].values.tolist())
    fill.update(sp['Symbol'].values.tolist())
    fill.update(sp['Name'].str.lower().values.tolist())
    fill.update(sp['connect'].values.tolist())
    fill.update(sp['rm_last'].values.tolist())
    fill.update(sp['rm_last_connect'].values.tolist())
    fill.update(sp['phrase_connect'].values.tolist())
    fill.update(sp['no_phrase_connect'].values.tolist())
    fill.update(
        ['us', 'standardpoor', 'america', 'exxon', 'wall_st', 'wall_street', 'wall st', 'wall street', 'u_s', 'j_j',
         'h_r', 'at_t', 'p_g', 'r_d', 'dow_jones', 'b_l', 'h_m', 's_p', 'amazon', 'exxon_mobil'])
    fill = list(filter(None, fill))
    print('length of total filter', len(fill))
    title_select = title.groupby(title.index).apply(list).apply(pd.Series).fillna('')
    for i in title_select.index:
        if len(list(filter(None, title_select.loc[i].values))) > 1:
            try:
                title.loc[i] = title.loc[i].apply(fil_sub, fill=fill)
            except AttributeError as e:
                print(i + ' ' + str(e))

    return title


def Ding_abstract(label, bigram, trigram, path='/Users/maobu/Dropbox/stock/data/ding/'):
    types = ['title', 'abstract']
    start = timeit.default_timer()
    print('start processing Ding_abstract data')
    article = defaultdict(list)
    with open('/home/huicheng/Documents/datas/yunke_embedding_v2/reuters_title_ner.json',
              "r") as data:  # title+ abstract + article
        title = pd.DataFrame(json.loads(line, strict=False) for line in data).set_index('date')
    
    with open('/home/huicheng/Documents/datas/yunke_embedding_v2/bloomberg_title_ner.json',
              "r") as data:  # title+ abstract + article
        bloomberg = pd.DataFrame(json.loads(line, strict=False) for line in data).set_index('date')
    title = title.append(bloomberg)
    
    print('original shape', title.shape)
    ##TODO don't do this if want to use character level
    title['title'] = title['title'].apply(lambda x: [x])
    title['abstract'] = title['abstract'].apply(lambda x: [x])
    ##TODO fill empty with nan,only use when using abstract and article
    # title = title.applymap(lambda x: np.nan if not x else x)
    ##TODO drop nan
    title = title.dropna(axis=0, how='any')  # drop the row that's incomplete
    title = title.sort_index()
    print('after dropping empty abstract and articles, first drop nan', title.shape)
    title = title.applymap(lambda x: '.'.join(x))
    title = title.replace(
        ['corrected', 'UPDATE\s\d-', "'s", 'update xxx', 'wrapup xxx', 'factbox', 'instant view', 'snap analysis',
         'exclusive',
         'timeline', 'highlights', 'correction', 'scenarios', 'analysis'], '', regex=True)  # becareful of using inplace
    title = title.replace(['\\bus\\b', '\\bu s\\b', "\\bu_s\\b"], 'america', regex=True)  # becareful of using inplace
    title = title.replace(['\\buk\\b', '\\bu k\\b', "\\bu_k\\b"], 'british', regex=True)  # becareful of using inplace
    title = title.replace(['\\bs p\\b', '\\bs_p\\b', "\\bs_p xxx\\b"], 'standardpoor',
                          regex=True)  # becareful of using inplace
    title = title.replace(['\ss\s'], ' ', regex=True)  # becareful of using inplace
    title = title.replace(
        ['\\bwo not\\b', '\\bbln\\b', '\\bmln\\b', '\\bpct\\b', '\\bs korea\\b', '\\bn korea\\b', '3rd', '2nd', '1st',
         '4th',
         '\\bq1\\b',
         '\\bq2\\b', '\\bq3\\b', '\\bq4\\b', '\\bwall street\\b', '\\bisnt\\b', '\\b1st_', '\\b2nd_', '\\b3rd_',
         '\\b4th_',
         '\\bfirst_', '\\bsecond_', '\\bthird_', '\\bfourth_'],
        ['would not', 'billion', 'million', 'percent', 'south_korea', 'north_korea', 'third', 'second', 'first',
         'fourth',
         'first quarter', 'second quarter', 'third quarter', 'fourth quarter', 'wall_street', 'is not', 'first ',
         'second ', 'third ', 'fourth ', 'first ', 'second ', 'third ', 'fourth '],
        regex=True)  # becareful of using inplace
    title = title.replace(['qtr', 'quarterly'], 'quarter', regex=True)  # becareful of using inplace
    title = title.replace(
        ['\\bj j\\b', '\\bh r\\b', '\\bat t\\b', '\\bp g\\b', '\\br d\\b', 'dow jones', '\\bdow\\b',
         '\\bb l\\b',
         '\\bh m\\b', ' _ ', '\\b\st\\b'],
        ['j_j', 'h_r', 'at_t', 'p_g', 'r_d', 'dow_jones', 'dow_jones', 'b_l', 'h_m', ' and ', 'street'],
        regex=True)  # becareful of using inplace
    title = title.replace(['\d+\S\d+', 'xxx xxx', 'xxx xxx xxx'], 'xxx',
                          regex=True)  # becareful of using inplace
    ##TODO only work on title
    print('finish dealing with manual replace')
    title['title'] = title['title'].apply(convert, args=(bigram, trigram, False, False, True))
    title = title.drop_duplicates(subset='title', keep='first')
    print('after dropping ' + 'title' + ' is {}'.format(title.shape))
    '''
    for i in types:
        # title[i] = title[i].str.replace('[{}]'.format(string.punctuation), ' ')
        title[i] = title[i].apply(convert, args=(bigram, trigram, False, False, False))
        #title = title.drop_duplicates(subset=[i], keep='first')
        print('after dropping ' + i + ' is {}'.format(title.shape))
    '''
    ##TODO change the news that's too short to nan, only use when processing abstract and article
    title['title'] = title['title'].apply(lambda x: x if len(x.split()) > 2 else np.nan)
    # title['abstract'] = title['abstract'].apply(lambda x: x if len(x) >= 100 else np.nan)
    # title['article'] = title['article'].apply(lambda x: x if len(x) >= 100 else np.nan)
    ##TODO filter the news
    print('start the filter process')
    #title['title'] = filter_news(title['title'], bigram, trigram)
    ##TODO drop nan
    title = title.dropna(axis=0, how='any')  # drop the row that's incomplete
    print('after dropping incomplete rows, second drop nan', title.shape)
    # title = title.applymap(lambda x: x.split('.'))
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
    print('before dropping the words that occurs less than 5 times', len(word_freqs))
    '''
    for key, value in list(word_freqs.items()):
        if value < 5:
            del word_freqs[key]
    '''
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


def convert(reviews, bigram, trigram, remove_stopwords=True, phrase=True, lemma=True):
    # letters_only = re.sub("[^a-zA-Z0-9]", " ", str(reviews))
    # words = list(filter(None, letters_only.lower().split()))
    words = word_tokenize(reviews)  # tokenize and remove punctuation
    if remove_stopwords:  # remove stopwords
        words = [w for w in words if not w in stopwords.words("english")]
    if phrase is True:
        words = trigram[bigram[words]]  # to phrase
    if lemma is True:
        words = lemmatize_sentence(words)  # lemma
    return " ".join(words)  # + ' .'


def main():
    # nltk.download('stopwords')
    print("start pre-processing the data")
    bigram = SaveLoad.load("big_phrase.pickle")
    trigram = SaveLoad.load("trig_phrase.pickle")
    label_one = pd.read_pickle("label_one.pickle")['2006-11-20':'2013-11-21']  # ['2006-11-20':'2013-11-21']
    path = 'ding_new_15/'
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
