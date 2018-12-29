import pickle as pkl
import gzip
import numpy
import random
import math
import pandas as pd


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator."""

    def __init__(self, source, label,
                 dict,
                 batch_size=32,
                 n_words=-1,
                 shuffle=True):
        self.source = pd.read_csv(source, header=None,low_memory=False,dtype='unicode').fillna("")
        self.label = pd.read_csv(label, header=None,low_memory=False,dtype='unicode').fillna("")
        with open(dict, 'rb') as f:
            self.dict = pkl.load(f)
        self.batch_size = batch_size
        self.n_words = n_words
        self.shuffle = shuffle
        self.end_of_data = False

        self.source_buffer = []
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
        label = []
        temp = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.label_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for i in range(self.k):
                try:
                    ss = list(filter(None, self.source.values[self.index + i]))
                    ll = self.label.values[self.index + i]
                except IndexError:  # out of length
                    break

                self.source_buffer.append(ss)
                self.label_buffer.append(int(ll))
            self.index += i + 1  # self.k

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
                _lbuf = [self.label_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.label_buffer = _lbuf
            for i in self.source_buffer:
                temp.append([j.strip().split() for j in i])
            self.source_buffer = temp
            '''if isinstance(self.source_buffer[0][0], list) is not True:  #check if it is a list
                for i in self.source_buffer:
                    temp.append([j.strip().split() for j in i])
                self.source_buffer = temp'''
        if len(self.source_buffer) == 0 or len(self.label_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            # actual work here
            '''for i in self.source_buffer:
                source_temp = []
                for j in i:  # len(source_buffer)       # read from source file and map to word index
                    j.insert(0, '_BOS_')
                    j.append('_EOS_')
                    ss = [self.dict[w] if w in self.dict else 1 for w in j]
                    if self.n_words > 0:
                        ss = [w if w < self.n_words else 1 for w in ss]
                        # read label
                    source_temp.append(ss)
                source.append(source_temp)
                label.append(self.label_buffer.pop(0))
                if len(source) >= self.batch_size or len(label) >= self.batch_size:
                    break
            del self.source_buffer[0:self.batch_size]'''
            while True:
                # read from source file and map to word index
                source_temp = []
                try:
                    j = self.source_buffer.pop(0)
                except IndexError:
                    break
                for i in j:
                    i.insert(0, '_BOS_')
                    i.append('_EOS_')
                    ss = [self.dict[w] if w in self.dict else 1 for w in i]
                    if self.n_words > 0:
                        ss = [w if w < self.n_words else 1 for w in ss]
                    source_temp.append(ss)
                # read label
                ll = self.label_buffer.pop(0)
                source.append(source_temp)
                label.append(ll)

                if len(source) >= self.batch_size or len(label) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(label) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, label


def main():
    train = TextIterator('../../data/ding/one_train_text.txt', '../../data/ding/one_train_label.txt',
                         dict='../../data/ding/vocab_cased.pickle',
                         n_words=43920,
                         batch_size=32, shuffle=True)
    for i, (x, y) in enumerate(train):
        print("train1", i)
    for i, (x, y) in enumerate(train):
        print("train2", i)


if __name__ == '__main__':
    main()
