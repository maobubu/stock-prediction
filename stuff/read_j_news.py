import json
import re
import tqdm
import collections
import csv
import tqdm
import numpy as np
import os
from multiprocessing import Process
import multiprocessing
import threading
import asyncio
import time
import functools
import timeit
import glob, os
from os import listdir
import string
import re
import json
import pandas as pd
from itertools import islice
import csv
import nltk
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
    return tokenize.tokenize(text.lower())# doesn't remove stop words
    #return [w for w in tokenize.tokenize(text.lower()) if not w in stopwords.words("english")]

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
    sente = ""
    for i in range(len(lemmas)):
        sente = sente + lemmas[i] + " "
    return sente


def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    return re.sub(strip_special_chars, " ", string.lower())


def cleanparent(string):
    string = re.sub(r"\(.*?\)", " ", string)
    return string


def cleannumber(string):
    nonum = ""
    word_list = list(string.split())
    for m in range(len(word_list)):
        try:
            word_list[m] = int(word_list[m])
            word_list[m] = "xxx"
            nonum = nonum + " " + word_list[m]
        except ValueError:
            nonum = nonum + " " + word_list[m]
            continue
    return nonum


file = open("/home/yunke/Desktop/data/reuters_news.json")
last = ""
output = open("/home/yunke/Desktop/Reuters.txt", "w")
training_file = open("/home/yunke/Desktop/reuters_bloomberg_train.json", "w")
bloom_file = open("/home/yunke/Desktop/data/bloomberg_news.json")
counter = 0


# READ REUTERS RAW NEWS INTO WORD2VEC INPUT FORMAT AND DO THE PREPROCCING
for line in tqdm.tqdm(file.readlines()):
    #no_num = ""
    dic = json.loads(line)
    raw_title = dic["title"].lower()  # get raw title
    raw_title = raw_title.replace("'s", "").replace("u.s.", "america").replace("american", "america")
    raw_title = raw_title.replace("update 1-", "").replace("update 2-", "").replace("update 3-", "").replace("update 4-", "")\
        .replace("update 5-", "").replace("update 6-", "").replace("update 7-", "").replace("update 8-", "")\
        .replace("update 9-", "").replace("update 10-", "")
    raw_title = raw_title.replace("wall st.", "wall street").replace("s&p 500", "standardpoor").replace("s&p", "standardpoor")
    raw_title = raw_title.replace("factbox:", "").replace("analysis:", "").replace("insight:", "").replace("advisory:", "")\
        .replace("bernanke:", "").strip(" ")
    no_pun_title = cleanSentences(raw_title)
    no_num_title = cleannumber(no_pun_title)
    if no_num_title == last:  # eliminate same news
        continue

    #  finish deal with title
    raw_abstract = dic["abstract"].lower()
    raw_abstract = cleanparent(raw_abstract)
    raw_abstract = raw_abstract.replace("'s", "").replace("u.s.", "america").replace("american", "america")
    raw_abstract = raw_abstract.replace("update 1-", "").replace("update 2-", "").replace("update 3-", "").replace("update 4-", "") \
        .replace("update 5-", "").replace("update 6-", "").replace("update 7-", "").replace("update 8-", "") \
        .replace("update 9-", "").replace("update 10-", "")
    raw_abstract = raw_abstract.replace("wall st.", "wall street").replace("s&p 500", "standardpoor").replace("s&p", "standardpoor")
    raw_abstract = raw_abstract.replace("factbox:", "").replace("analysis:", "").replace("insight:", "").replace("advisory:", "") \
        .replace("bernanke:", "").strip(" ")
    no_pun_abstract = cleanSentences(raw_abstract)
    no_num_abstract = cleannumber(no_pun_abstract)

    #  finish deal with abstract
    raw_article = dic["article"].lower()
    raw_article = cleanparent(raw_article)
    raw_article = raw_article.replace("'s", "").replace("u.s.", "america").replace("american", "america")
    raw_article = raw_article.replace("update 1-", "").replace("update 2-", "").replace("update 3-", "").replace("update 4-",
                                                                                                          "") \
        .replace("update 5-", "").replace("update 6-", "").replace("update 7-", "").replace("update 8-", "") \
        .replace("update 9-", "").replace("update 10-", "")
    raw_article = raw_article.replace("wall st.", "wall street").replace("s&p 500", "standardpoor").replace("s&p", "standardpoor")
    raw_article = raw_article.replace("factbox:", "").replace("analysis:", "").replace("insight:", "").replace("advisory:", "") \
        .replace("bernanke:", "").strip(" ")
    no_pun_article = cleanSentences(raw_article)
    no_num_article = cleannumber(no_pun_article)
    #  finish deal with article

    #print(no_num_title.strip(), end=" ", file=output)
    #print(no_num_abstract.strip(), end=" ", file=output)
    #print(no_num_article.strip(), end=" ", file=output)
    #print(dic["date"], "\t", no_num.strip(" "), file=output)
    data = {"date": "", "title": "", "abstract": "", "article": ""}
    data["date"] = dic["date"]
    data["title"] = no_num_title.strip()
    data["abstract"] = no_num_abstract.strip()
    data["article"] = no_num_article.strip()
    o_data = collections.OrderedDict(data)
    json_o = json.dump(o_data, training_file)
    print("\r", file=training_file)
    last = no_num_title
    counter = counter + 1
    #if counter == 50:



# READ BLOOMBERG RAW NEWS INTO WORD2VEC INPUT FORMAT AND DO THE PREPROCCING
for line in tqdm.tqdm(bloom_file.readlines()):
    # no_num = ""
    dic = json.loads(line)
    raw_title = dic["title"].lower()  # get raw title
    raw_title = raw_title.replace("'s", "").replace("u.s.", "america").replace("american", "america")
    raw_title = raw_title.replace("update 1-", "").replace("update 2-", "").replace("update 3-", "").replace(
        "update 4-", "") \
        .replace("update 5-", "").replace("update 6-", "").replace("update 7-", "").replace("update 8-", "") \
        .replace("update 9-", "").replace("update 10-", "")
    raw_title = raw_title.replace("wall st.", "wall street").replace("s&p 500", "standardpoor").replace("s&p",
                                                                                                        "standardpoor")
    raw_title = raw_title.replace("factbox:", "").replace("analysis:", "").replace("insight:", "").replace("advisory:",
                                                                                                           "") \
        .replace("bernanke:", "").strip(" ")
    no_pun_title = cleanSentences(raw_title)
    no_num_title = cleannumber(no_pun_title)
    if no_num_title == last:  # eliminate same news
        continue

    #  finish deal with title
    raw_abstract = dic["abstract"].lower()
    raw_abstract = cleanparent(raw_abstract)
    raw_abstract = raw_abstract.replace("'s", "").replace("u.s.", "america").replace("american", "america")
    raw_abstract = raw_abstract.replace("update 1-", "").replace("update 2-", "").replace("update 3-", "").replace(
        "update 4-", "") \
        .replace("update 5-", "").replace("update 6-", "").replace("update 7-", "").replace("update 8-", "") \
        .replace("update 9-", "").replace("update 10-", "")
    raw_abstract = raw_abstract.replace("wall st.", "wall street").replace("s&p 500", "standardpoor").replace("s&p",
                                                                                                              "standardpoor")
    raw_abstract = raw_abstract.replace("factbox:", "").replace("analysis:", "").replace("insight:", "").replace(
        "advisory:", "") \
        .replace("bernanke:", "").strip(" ")
    no_pun_abstract = cleanSentences(raw_abstract)
    no_num_abstract = cleannumber(no_pun_abstract)

    #  finish deal with abstract
    raw_article = dic["article"].lower()
    raw_article = cleanparent(raw_article)
    raw_article = raw_article.replace("'s", "").replace("u.s.", "america").replace("american", "america")
    raw_article = raw_article.replace("update 1-", "").replace("update 2-", "").replace("update 3-", "").replace(
        "update 4-",
        "") \
        .replace("update 5-", "").replace("update 6-", "").replace("update 7-", "").replace("update 8-", "") \
        .replace("update 9-", "").replace("update 10-", "")
    raw_article = raw_article.replace("wall st.", "wall street").replace("s&p 500", "standardpoor").replace("s&p",
                                                                                                            "standardpoor")
    raw_article = raw_article.replace("factbox:", "").replace("analysis:", "").replace("insight:", "").replace(
        "advisory:", "") \
        .replace("bernanke:", "").strip(" ")
    no_pun_article = cleanSentences(raw_article)
    no_num_article = cleannumber(no_pun_article)
    #  finish deal with article

    #print(no_num_title.strip(), end=" ", file=output)
    #print(no_num_abstract.strip(), end=" ", file=output)
    #print(no_num_article.strip(), end=" ", file=output)
    # print(dic["date"], "\t", no_num.strip(" "), file=output)
    data = {"date": "", "title": "", "abstract": "", "article": ""}
    data["date"] = dic["date"]
    data["title"] = no_num_title.strip()
    data["abstract"] = no_num_abstract.strip()
    data["article"] = no_num_article.strip()
    o_data = collections.OrderedDict(data)
    json_o = json.dump(o_data, training_file)
    print("\r", file=training_file)
    last = no_num_title
    counter = counter + 1

file.close()
#print(counter)

