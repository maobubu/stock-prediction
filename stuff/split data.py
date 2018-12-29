import pandas as pd
import glob, os, json
from gensim.models import Phrases
import string
from gensim.utils import SaveLoad
from gensim.models.phrases import Phraser
from gensim.models.word2vec import Word2Vec
from datetime import datetime
from datetime import timedelta
from gensim.models.word2vec import Word2Vec
from collections import defaultdict
from gensim.models import KeyedVectors


def main():
    denominator, numerator = 0, 0
    df = pd.read_csv("SP.csv")
    #df = pd.read_table("AAPL.csv")
    df = df.set_index('Date')
    label_one = df.Close.diff(1).fillna(0)
    label_seven = df.Close.diff(7).fillna(0)
    label_month = df.Close.diff(30).fillna(0)
    label_one[label_one > 0] = 1
    label_one[label_one <= 0] = 0
    label_seven[label_one > 0] = 1
    label_seven[label_one <= 0] = 0
    label_month[label_one > 0] = 1
    label_month[label_one <= 0] = 0
    label_one = label_one["2006-10-01":]
    label_seven = label_seven#["2006-10-20":"2013-11-21"]
    label_month = label_month#["2006-10-20":]
    label_one.to_pickle("data/label_one.pickle")
    label_seven.to_pickle("data/label_seven.pickle")
    label_month.to_pickle("data/label_month.pickle")
    list(label_one.index)
    os.chdir('/home/huicheng/Documents/datas/reuters_new/reuters_new')
    sentence_stream = []
    for i, file in enumerate(glob.glob("*.json")):
        with open(file, "r") as data:
            D = pd.DataFrame(json.loads(line) for line in data)
        D = D.set_index('date')
        D=D.replace(['UPDATE\s\d-', "'s"], '', regex=True)
        D=D.replace(['\d+\S\d+','\d+'], 'xxx', regex=True)
        D['title'] = D['title'].str.replace('[{}]'.format(string.punctuation), ' ')
        #D['title'] = D['title'].str.replace('[{}]'.format(string.digits), '_NUM_ ')
        D = D.drop_duplicates(subset=['title'], keep='first')
        sentence_stream += [doc.strip().lower().split(" ") for doc in D['title'].values]
        D.index = pd.to_datetime(D.index, format='%Y%m%d')  # Set the indix to a datetime
        test = D.loc[:'2015-06-18']  # split train and test
        train = D.loc['2015-06-17':]
        numerator += test.shape[0]
        denominator += D.shape[0]
        D.to_pickle("/home/huicheng/PycharmProjects/stock/pickle/" + os.path.splitext(file)[0] + '.pickle')
        # train.to_pickle("/home/huicheng/PycharmProjects/bilstm/train/" + os.path.splitext(file)[0] + '.pickle')
        # test.to_pickle("/home/huicheng/PycharmProjects/bilstm/test/" + os.path.splitext(file)[0] + '.pickle')
    print(numerator, denominator)
    os.chdir("/home/huicheng/PycharmProjects/stock/")
    # TODO phrase
    '''phrases = Phrases(sentence_stream)
    bigram = Phraser(phrases)
    print(list(bigram[sentence_stream]))
    bigram.save("phrase.pickle")
    # TODO word2vec
    model = Word2Vec(bigram[sentence_stream], size=100, window=5, min_count=5, workers=4)
    model.save("word2vec.pickle")
    print(model.wv['computer'])
    print(model.wv.most_similar(positive=['wall_st', 'u_s'], negative=['apple']))'''


if __name__ == '__main__':
    main()
