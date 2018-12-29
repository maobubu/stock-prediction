#!usr/bin/env python3
from pycorenlp import StanfordCoreNLP

import numpy as np
import time
# import word2vec
import string
from collections import defaultdict
import pickle
import json
import pandas as pd

nlp = StanfordCoreNLP('http://localhost:9000')
relations = dict()


def open_ie(input, relations=relations, nlp=nlp):
    date = input['date']
    outputs = defaultdict(list)
    start_time = time.time()
    for z, i in enumerate(list(filter(None,input.values[1:]))):
        output = nlp.annotate(i, properties={
            'annotators': 'openie',
            'outputFormat': 'json'
        })
        try:
            j = list(filter(None, output['sentences'][0]['openie']))
            if str(len(j)) not in relations:
                relations[str(len(j))] = 0
            relations[str(len(j))] += 1
            if j:
                outputs[date].append(j)
            print('text {}: {} sec'.format(z, time.time() - start_time))
        except TypeError as e:
            print(str(e))
            continue
    with open('openie.json', 'a+') as f:
        if outputs:
            json.dump(outputs, f)
            f.write('\r')


def main():
    a = pd.read_table("bloomberg_news_title.txt", names=['date', 'title']).set_index('date')
    b = pd.read_table("reuters_news_title.txt", names=['date', 'title']).set_index('date')
    title = a.append(b)
    # title = title.sort_index()
    title = title['title'].groupby(title.index).apply(list).apply(pd.Series).fillna('')  # group together
    title = title.reset_index()
    title.apply(open_ie, axis=1)
    print(relations)
    '''
    with open('openie.p', 'wb') as outfile:
        pickle.dump(outputs, outfile)
    '''


if __name__ == '__main__':
    main()
