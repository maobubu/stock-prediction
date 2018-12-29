import json
import pandas as pd
import re, os, glob
import numpy as np
from collections import defaultdict
import nltk
import string
from gensim.models import Phrases
from gensim.utils import SaveLoad
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords  # Import the stop word list
#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from datetime import timedelta
import timeit
import sys
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
#from nltk import pos_tag, word_tokenize
from nltk.tag import PerceptronTagger
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
    return tokenize.tokenize(text.lower())#doesn't remove stopwords
    #return [w for w in tokenize.tokenize(text.lower()) if not w in stopwords.words("english")]

def lemmatize_sentence(sentence, neverstem=False, keepWordPOS=False, 
                       tokenizer=word_tokenize, postagger=pos_tag, 
                       lemmatizer=wnl, stemmer=porter):
    words, lemmas, poss = [], [], []
    for word, pos in postagger(sentence):#change tokenizer(sentence) to sentence
        pos = penn2morphy(pos)
        lemmas.append(lemmatize(word.lower(), pos, neverstem,
                                lemmatizer, stemmer))
        poss.append(pos)
        words.append(word)
    if keepWordPOS:
        return words, lemmas, [None if i == '' else i for i in poss]
    return lemmas

def Ding_abstract(label,bigram,trigram,types='title'):
    start = timeit.default_timer()
    print('start processing Ding_abstract data')
    article = defaultdict(list)
    with open('/home/jialong/Documents/phrase_embedding/j_news.json', "r") as data:    # title+ abstract + article
        title = pd.DataFrame(json.loads(line) for line in data).set_index('date')
    title=title.replace(['UPDATE\s\d-', "'s"], '', regex=True)
    title=title.replace(['\d+\S\d+','\d+'], 'xxx', regex=True)
    title[types] = title[types].str.replace('[{}]'.format(string.punctuation), ' ')
    title = title.drop_duplicates(subset=[types], keep='first')
    for j in label.index:
        try:
            day = (datetime.strptime(j, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            article[j].extend(set(title.loc[day, types].values))
        except (AttributeError, KeyError, TypeError) as e:
            continue
    train_ding = split(article, label, bigram, trigram)
    length = train_ding.shape[0]
    train = train_ding.iloc[0:int(length * 0.8), :]
    validate = train_ding.iloc[int(length * 0.8):int(length * 0.9), :]
    test = train_ding.iloc[int(length * 0.9):-1, :]
    stop = timeit.default_timer()
    print("run time for ding:", stop - start)
    #os.chdir('/Users/maobu/Dropbox/stock/data/ding/')
    os.chdir('/home/jialong/Documents/phrase_embedding/data/yunke_'+types+'/')
    train.to_csv("one_train.csv", index=False, encoding='utf-8')
    df_train = np.split(train, [1], axis=1)
    df_train[1].to_csv('one_train_text.txt', header=None, index=None, encoding='utf-8')
    df_train[0].to_csv('one_train_label.txt', header=None, index=None, encoding='utf-8')
    validate.to_csv("one_validate.csv", index=False, encoding='utf-8')
    df_validate = np.split(validate, [1], axis=1)
    df_validate[1].to_csv('one_validate_text.txt', header=None, index=None, encoding='utf-8')
    df_validate[0].to_csv('one_validate_label.txt', header=None, index=None, encoding='utf-8')
    test.to_csv("one_test.csv", index=False, encoding='utf-8')
    df_test = np.split(test, [1], axis=1)
    df_test[1].to_csv('one_test_text.txt', header=None, index=None, encoding='utf-8')
    df_test[0].to_csv('one_test_label.txt', header=None, index=None, encoding='utf-8')

def Ding(label, bigram, trigram,types='title'):
    start = timeit.default_timer()
    print('start processing Ding data')
    article = defaultdict(list)
    title = pd.read_table('reuters_news_title.txt', names=["Date", 'title']).set_index('Date')
    #tt=pd.read_table('bloomberg_news_title.txt',names = ["Date", 'title']).set_index('Date')
    #title=title.append(tt)
    title=title.replace(['UPDATE\s\d-', "'s"], '', regex=True)
    title=title.replace(['\d+\S\d+','\d+'], 'xxx', regex=True)
    #title['title'] = title['title'].str.replace('[{}]'.format(string.digits), '_NUM_ ')
    title[types] = title[types].str.replace('[{}]'.format(string.punctuation), ' ')
    title = title.drop_duplicates(subset=[types], keep='first')
    for j in label.index:
        try:
            day = (datetime.strptime(j, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            article[j].extend(set(title.loc[day, types].values))
        except (AttributeError, KeyError, TypeError) as e:
            continue
    train_ding = split(article, label, bigram, trigram)
    length = train_ding.shape[0]
    train = train_ding.iloc[0:int(length * 0.8), :]
    validate = train_ding.iloc[int(length * 0.8):int(length * 0.9), :]
    test = train_ding.iloc[int(length * 0.9):-1, :]
    stop = timeit.default_timer()
    print("run time for ding:", stop - start)
    os.chdir('/Users/maobu/Dropbox/stock/data/ding/')
    train.to_csv("one_train.csv", index=False, encoding='utf-8')
    df_train = np.split(train, [1], axis=1)
    df_train[1].to_csv('one_train_text.txt', header=None, index=None, encoding='utf-8')
    df_train[0].to_csv('one_train_label.txt', header=None, index=None, encoding='utf-8')
    validate.to_csv("one_validate.csv", index=False, encoding='utf-8')
    df_validate = np.split(validate, [1], axis=1)
    df_validate[1].to_csv('one_validate_text.txt', header=None, index=None, encoding='utf-8')
    df_validate[0].to_csv('one_validate_label.txt', header=None, index=None, encoding='utf-8')
    test.to_csv("one_test.csv", index=False, encoding='utf-8')
    df_test = np.split(test, [1], axis=1)
    df_test[1].to_csv('one_test_text.txt', header=None, index=None, encoding='utf-8')
    df_test[0].to_csv('one_test_label.txt', header=None, index=None, encoding='utf-8')


def convert(reviews, bigram, trigram, remove_stopwords=True):
    #letters_only = re.sub("[^a-zA-Z0-9]", " ", str(reviews))
    #words = list(filter(None, letters_only.lower().split()))
    words= word_tokenize(reviews)#tokenize and remove punctuation
    if remove_stopwords:#remove stopwords
        words = [w for w in words if not w in stopwords.words("english")]
    words = trigram[bigram[words]]#to phrase
    words= lemmatize_sentence(words)#lemma
    return " ".join(words) + ' .'


def split(data, label, bigram, trigram):
    data_clean = []
    lab = []
    date = []
    for key, value in data.items():
        for j in set(value):
            try:
                lab.append(label[key])
                data_clean.append(convert(j, bigram, trigram, True))
            except (KeyError, TypeError) as e:
                continue
    print(len(data_clean))
    ll = pd.DataFrame({'label': lab}, dtype='int32')
    d = pd.DataFrame({'title': data_clean})  # put the convert words into a new Dataframe
    final = pd.merge(ll, d, left_index=True, right_index=True)  # merge two list
    return final


def maobu(label, d, article, abstract, days, add=False):
    for j in label.index:
        # for i in range(1, days + 1):
        try:
            day = (datetime.strptime(j, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')
            article[j].extend(set(d.loc[day, "title"].values))
            if add:
                abstract[j].extend(set(d.loc[day, "abstract"].values))
        except (AttributeError, KeyError, TypeError) as e:
            continue


def main():
    arg1 = sys.argv[1]
    one_train, abstract_train, seven_train, month_train = defaultdict(list), defaultdict(list), defaultdict(
        list), defaultdict(list)
    one_test, seven_test, month_test = defaultdict(list), defaultdict(list), defaultdict(list)
    # nltk.download('stopwords')
    print("start pre-processing the data")
    bigram = SaveLoad.load("data/phrase_xxx/big_phrase.pickle")
    trigram = SaveLoad.load("data/phrase_xxx/trig_phrase.pickle")
    label_one = pd.read_pickle("data/label_one_new.pickle")
    label_seven = pd.read_pickle("data/label_seven.pickle")
    label_month = pd.read_pickle("data/label_month.pickle")
    print("starting the training selecting phase")
    Ding(label_one, bigram, trigram,types=arg1)
    #Ding_abstract(label_one, bigram, trigram,types=str(arg1))
    '''os.chdir('/home/huicheng/PycharmProjects/stock/pickle')
    subfolder_list = glob.glob('*.pickle')
    pbar = tqdm(total=len(subfolder_list))
    for i, file in enumerate(glob.glob("*.pickle")):
        D = pd.read_pickle(file)
        pbar.set_description('processing number:{} name:{}'.format(i, file))
        pbar.update(1)
        maobu(label_one, D, one_train, abstract_train, 1, add=False)  # add abstract or not
        # maobu(label_seven, D, seven_train, 7)
        # maobu(label_month, D, month_train, 30)
    pbar.close()
    start = timeit.default_timer()
    train_one = split(one_train, label_one, bigram, trigram)
    length = train_one.shape[0]
    train = train_one.iloc[0:int(length * 0.8), :]
    validate = train_one.iloc[int(length * 0.8):int(length * 0.9), :]
    test = train_one.iloc[int(length * 0.9):-1, :]
    # train_seven = split(seven_train, label_seven)
    # train_month = split(month_train, label_month)
    stop = timeit.default_timer()
    print("run time for training:", stop - start)
    os.chdir('/home/huicheng/PycharmProjects/stock/data/our')
    train.to_csv("one_train.csv", index=False, encoding='utf-8')
    df_train = np.split(train, [1], axis=1)
    df_train[1].to_csv('one_train_text.txt', header=None, index=None, encoding='utf-8')
    df_train[0].to_csv('one_train_label.txt', header=None, index=None, encoding='utf-8')
    validate.to_csv("one_validate.csv", index=False, encoding='utf-8')
    df_validate = np.split(validate, [1], axis=1)
    df_validate[1].to_csv('one_validate_text.txt', header=None, index=None, encoding='utf-8')
    df_validate[0].to_csv('one_validate_label.txt', header=None, index=None, encoding='utf-8')
    test.to_csv("one_test.csv", index=False, encoding='utf-8')
    df_test = np.split(test, [1], axis=1)
    df_test[1].to_csv('one_test_text.txt', header=None, index=None, encoding='utf-8')
    df_test[0].to_csv('one_test_label.txt', header=None, index=None, encoding='utf-8')

    # TODO split the abstract
    train_abstract = split(abstract_train, label_one, bigram, trigram)
    length = train_abstract.shape[0]
    train = train_abstract.iloc[0:int(length * 0.8), :]
    validate = train_abstract.iloc[int(length * 0.8):int(length * 0.9), :]
    test = train_abstract.iloc[int(length * 0.9):-1, :]
    # train_seven = split(seven_train, label_seven)
    # train_month = split(month_train, label_month)
    stop = timeit.default_timer()
    print("run time for training2:", stop - start)
    os.chdir('/home/huicheng/PycharmProjects/stock/data/our_abstract')
    train.to_csv("one_train.csv", index=False, encoding='utf-8')
    df_train = np.split(train, [1], axis=1)
    df_train[1].to_csv('one_train_text.txt', header=None, index=None, encoding='utf-8')
    df_train[0].to_csv('one_train_label.txt', header=None, index=None, encoding='utf-8')
    validate.to_csv("one_validate.csv", index=False, encoding='utf-8')
    df_validate = np.split(validate, [1], axis=1)
    df_validate[1].to_csv('one_validate_text.txt', header=None, index=None, encoding='utf-8')
    df_validate[0].to_csv('one_validate_label.txt', header=None, index=None, encoding='utf-8')
    test.to_csv("one_test.csv", index=False, encoding='utf-8')
    df_test = np.split(test, [1], axis=1)
    df_test[1].to_csv('one_test_text.txt', header=None, index=None, encoding='utf-8')
    df_test[0].to_csv('one_test_label.txt', header=None, index=None, encoding='utf-8')
    '''

if __name__ == '__main__':
    main()
