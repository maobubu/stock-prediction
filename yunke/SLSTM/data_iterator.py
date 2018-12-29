import pickle as pkl
import gzip
import numpy
import random
import math
import pandas as pd
from datetime import datetime
from datetime import timedelta


def delay(j, day):
    return (datetime.strptime(j, '%Y-%m-%d') - timedelta(days=day)).strftime('%Y-%m-%d')


class TextIterator:
    """Simple Bitext iterator."""

    def __init__(self, source, label,
                 dict, delay1=3, delay2=7, types='title',
                 batch_size=32,
                 n_words=-1,
                 cut_word=False, cut_sentence=False, cut_news=False,
                 shuffle=True, shuffle_sentence=True):  # delay means how many days over the past
        self.source = pd.read_csv(source).set_index('date')
        self.source = self.source[types].groupby(self.source.index).apply(list).apply(pd.Series).fillna(
            '')  # group together
        self.label = pd.read_csv(label).set_index('Date')
        with open(dict, 'rb') as f:
            self.dict = pkl.load(f)
        self.batch_size = batch_size
        self.n_words = n_words
        self.shuffle = shuffle
        self.shuffle_sentence = shuffle_sentence
        self.delay1 = delay1
        self.delay2 = delay2
        self.types = types
        self.end_of_data = False
        self.cut_word = cut_word if cut_word else float('inf')  # cut the word
        self.cut_news = cut_news if cut_news else None  # cut the sentence
        self.cut_sentence = cut_sentence if cut_sentence else None
        self.source_buffer = []
        self.source_d1_buffer = []
        self.source_d2_buffer = []
        self.label_buffer = []
        self.k = batch_size * 20
        self.index = 0

    def __iter__(self):
        return self

    def reset(self):
        # self.source.seek(0)
        # self.label.seek(0)
        self.index = 0

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        source_d1 = []
        source_d2 = []
        label = []
        temp = []
        tempd1 = []
        tempd2 = []
        # day = (datetime.strptime(j, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.label_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for j, i in enumerate(self.label.index.values[self.index:self.index + self.k]):  # j for count i for value
                try:
                    ss, d1, d2 = [], [], []
                    for k in list(filter(None, self.source.loc[delay(i, 1)].values[:self.cut_news])):
                        sentence = []
                        for l in k.split(' . ')[:self.cut_sentence]:
                            sentence.append(l) if self.cut_word > len(l.split()) > 0 else None
                        ss.append(sentence)
                    for m in self.source.loc[delay(i, self.delay1):delay(i, 1 + 1)].values:
                        day = []
                        for k in list(filter(None, m[:self.cut_news])):
                            sentence = []
                            for l in k.split(' . ')[:self.cut_sentence]:
                                sentence.append(l) if self.cut_word > len(l.split()) > 0 else None
                            day.append(sentence)
                        d1.append(list(filter(None, day)))
                        #assert len(d1) == self.delay1 - 1, 'd1 Missing some day!'
                    for m in self.source.loc[delay(i, self.delay2):delay(i, self.delay1 + 1)].values:
                        day = []
                        for k in list(filter(None, m[:self.cut_news])):
                            sentence = []
                            for l in k.split(' . ')[:self.cut_sentence]:
                                sentence.append(l) if self.cut_word > len(l.split()) > 0 else None
                            day.append(sentence)
                        d2.append(list(filter(None, day)))
                        #assert len(d2) == self.delay2 - self.delay1 - 1, 'd2 Missing some day!'
                except KeyError as e:  # out of length
                    print(i + ' ' + str(e))
                    continue

                ll = self.label.loc[i].values
                self.source_buffer.append(list(filter(None, ss)))
                self.source_d1_buffer.append(d1)
                self.source_d2_buffer.append(d2)
                self.label_buffer.append(int(ll))
            if 'j' in locals():
                self.index += j + 1
            ##TODO delete useless

            if self.shuffle:
                # sort by target buffer
                tlen = numpy.array([len(t) for t in self.source_buffer])
                tidx = tlen.argsort()
                # argsort the index from low to high
                # shuffle mini-batch
                tindex = []
                ##Todo shuffle
                small_index = list(range(int(math.ceil(len(tidx) * 1. / self.batch_size))))
                random.shuffle(small_index)
                for i in small_index:
                    if (i + 1) * self.batch_size > len(tidx):
                        tindex.extend(tidx[i * self.batch_size:])
                    else:
                        tindex.extend(tidx[i * self.batch_size:(i + 1) * self.batch_size])
                tidx = tindex

                _sbuf = [self.source_buffer[i] for i in tidx]
                _d1buf = [self.source_d1_buffer[i] for i in tidx]
                _d2buf = [self.source_d2_buffer[i] for i in tidx]
                _lbuf = [self.label_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.source_d1_buffer = _d1buf
                self.source_d2_buffer = _d2buf
                self.label_buffer = _lbuf
                ##TODO delete useless
                del _sbuf, _d1buf, _d2buf, _lbuf
            for i, d1, d2 in zip(self.source_buffer, self.source_d1_buffer, self.source_d2_buffer):
                dd1, dd2, ss1, = list(), list(), list()
                for z in i:  # every news in one day
                    ss1.append([j.strip().split() for j in z])  # every sentence in one news, split words,save to array
                temp.append(ss1)
                for day in d1:  # every day
                    ddd1 = []
                    for z in day:  # every news in one day
                        ddd1.append([j.strip().split() for j in z])  # every sentence in one news
                    dd1.append(ddd1)
                tempd1.append(dd1)
                for day in d2:
                    ddd2 = []
                    for z in day:
                        ddd2.append([j.strip().split() for j in z])
                    dd2.append(ddd2)
                tempd2.append(dd2)
                # tempd2.append([j.strip().split() for day in d2 for j in day])
            self.source_buffer = temp
            self.source_d1_buffer = tempd1
            self.source_d2_buffer = tempd2
            ##TODO delete useless
            del temp, tempd1, tempd2
            ##TODO check if it contains enough day's new
            '''
            for j, i in enumerate(self.source_d1_buffer):
                if len(i) != self.delay1 - 1:
                    print(j)
            for j, i in enumerate(self.source_d2_buffer):
                if len(i) != self.delay2 - self.delay1:
                    print(j)
            '''
            ##TODO #check if it is a list
            '''
            if isinstance(self.source_buffer[0][0], list) is not True:  
                for i in self.source_buffer:
                    temp.append([j.strip().split() for j in i])
                self.source_buffer = temp
            '''
        if len(self.source_buffer) == 0 or len(self.label_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            # actual work here
            while True:
                # read from source file and map to word index
                source_temp, source_d1_temp, source_d2_temp = [], [], []
                try:
                    j = self.source_buffer.pop(0)  # 1 day before
                    d1j = self.source_d1_buffer.pop(0)  # delay1 day before
                    d2j = self.source_d2_buffer.pop(0)  # delay2 day before
                except IndexError:
                    break
                if self.shuffle_sentence:
                    numpy.random.shuffle(j)
                for z in j:  # each news
                    ss_temp = []
                    for i in z:  # each sentence
                        # i.insert(0, '_BOS_')
                        # i.append('_EOS_')
                        ss = [self.dict[w] if w in self.dict else 1 for w in i]  # 1 means _UNK_
                        if self.n_words > 0:
                            ss = [w if w < self.n_words else 1 for w in ss]  # 1 means _UNK_
                        ss_temp.append(ss)
                    source_temp.append(ss_temp)
                for a in d1j:  # each day
                    if self.shuffle_sentence:
                        numpy.random.shuffle(a)
                    _sd1 = []
                    for z in a:  # each news
                        d1_temp = []
                        for i in z:  # each sentence
                            # i.insert(0, '_BOS_')
                            # i.append('_EOS_')
                            ss = [self.dict[w] if w in self.dict else 1 for w in i]  # 1 means _UNK_
                            if self.n_words > 0:
                                ss = [w if w < self.n_words else 1 for w in ss]  # 1 means _UNK_
                            d1_temp.append(ss)
                        _sd1.append(d1_temp)
                    source_d1_temp.append(_sd1)
                for a in d2j:  # deal with delay2 each day
                    if self.shuffle_sentence:
                        numpy.random.shuffle(a)
                    _sd2 = []
                    for z in a:  # each news
                        d2_temp = []
                        for i in z:  # each sentence
                            # i.insert(0, '_BOS_')
                            # i.append('_EOS_')
                            ss = [self.dict[w] if w in self.dict else 1 for w in i]  # 1 means _UNK_
                            if self.n_words > 0:
                                ss = [w if w < self.n_words else 1 for w in ss]  # 1 means _UNK_
                            d2_temp.append(ss)
                        _sd2.append(d2_temp)
                    source_d2_temp.append(_sd2)
                # read label
                ll = self.label_buffer.pop(0)
                # assert len(self.source_buffer) == len(self.label_buffer), 'Buffer size mismatch!'
                source.append(source_temp)
                source_d1.append(source_d1_temp)
                source_d2.append(source_d2_temp)
                label.append(ll)
                ##TODO delete useless
                ##del source_temp, source_d1_temp, source_d2_temp

                if len(source) >= self.batch_size or len(source_d1) >= self.batch_size or len(
                        source_d2) >= self.batch_size or len(label) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(label) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, source_d1, source_d2, label


def main():
    train = TextIterator('ding_new_6/train.csv',
                         'ding_new_6/train_label.csv',
                         dict='ding_new_6/vocab_cased_title.pickle',
                         delay1=3,
                         delay2=7,
                         types='article',
                         n_words=43920,
                         batch_size=32, cut_word=False, cut_sentence=False, cut_news=False,
                         shuffle=False, shuffle_sentence=False)  # cut word: max length of the words in sentence
    validate = TextIterator('ding_new_6/validate.csv',
                            'ding_new_6/validate_label.csv',
                            dict='ding_new_6/vocab_cased_title.pickle',
                            delay1=3,
                            delay2=7,
                            types='article',
                            n_words=43920,
                            batch_size=32, cut_word=False, cut_sentence=False, cut_news=False,
                            shuffle=False, shuffle_sentence=False)  # cut word: max length of the words in sentence
    test = TextIterator('ding_new_6/test.csv',
                        'ding_new_6/test_label.csv',
                        dict='ding_new_6/vocab_cased_title.pickle',
                        delay1=3,
                        delay2=7,
                        types='article',
                        n_words=43920,
                        batch_size=32, cut_word=False, cut_sentence=False, cut_news=False,
                        shuffle=False, shuffle_sentence=False)  # cut word: max length of the words in sentence
    # cut news: max news number per day
    for i, (x, xd1, xd2, y) in enumerate(train):
        print("train", i, 'length', len(x))
    for i, (x, xd1, xd2, y) in enumerate(validate):
        print("validate", i, 'length', len(x))
    for i, (x, xd1, xd2, y) in enumerate(test):
        print("test", i, 'length', len(x))


if __name__ == '__main__':
    main()
