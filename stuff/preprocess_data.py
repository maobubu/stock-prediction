#!/usr/bin/python3
import sys
import os
import numpy
import pickle as pkl

from collections import OrderedDict

dic = {'entailment': '0', 'neutral': '1', 'contradiction': '2'}

def build_dictionary(filepaths, dst_path, lowercase=False):
    word_freqs = OrderedDict()
    for filepath in filepaths:
        print('Processing', filepath)
        with open(filepath, 'r') as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1

    words = list(word_freqs.keys())
    freqs = list(word_freqs.values())

    sorted_idx = numpy.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    worddict = OrderedDict()
    worddict['_PAD_'] = 0 # default, padding 
    worddict['_UNK_'] = 1 # out-of-vocabulary
    worddict['_BOS_'] = 2 # begin of sentence token
    worddict['_EOS_'] = 3 # end of sentence token

    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii + 4

    with open(dst_path, 'wb') as f:
        pkl.dump(worddict, f)

    print('Dict size', len(worddict))
    print('Done')


def build_sequence(filepath, dst_dir):
    filename = os.path.basename(filepath)
    print(filename)
    len_p = []
    len_h = []
    with open(filepath) as f, \
         open(os.path.join(dst_dir, 'premise_%s'%filename), 'w') as f1, \
         open(os.path.join(dst_dir, 'hypothesis_%s'%filename), 'w') as f2,  \
         open(os.path.join(dst_dir, 'label_%s'%filename), 'w') as f3:
        next(f) # skip the header row
        for line in f:
            sents = line.strip().split('\t')
            if sents[0] is '-':
                continue

            words_in = sents[1].strip().split(' ')
            words_in = [x for x in words_in if x not in ('(',')')]
            f1.write(' '.join(words_in) + '\n')
            len_p.append(len(words_in))

            words_in = sents[2].strip().split(' ')
            words_in = [x for x in words_in if x not in ('(',')')]
            f2.write(' '.join(words_in) + '\n')
            len_h.append(len(words_in))

            f3.write(dic[sents[0]] + '\n')

    print('max min len premise', max(len_p), min(len_p))
    print('max min len hypothesis', max(len_h), min(len_h))


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing maobu dataset')
    print('=' * 80)
    base_dir = os.path.dirname(os.path.realpath(__file__))
    #dst_dir = os.path.join(base_dir, 'data')
    dst_dir = os.path.join(base_dir, 'data/ding')
    #snli_dir = os.path.join(base_dir, 'data')
    make_dirs([dst_dir])

    '''build_sequence(os.path.join(snli_dir, 'snli_1.0_dev.txt'), dst_dir)
    build_sequence(os.path.join(snli_dir, 'snli_1.0_test.txt'), dst_dir)
    build_sequence(os.path.join(snli_dir, 'snli_1.0_train.txt'), dst_dir)'''

    build_dictionary([os.path.join(dst_dir, 'one_train_text.txt'), 
                      os.path.join(dst_dir, 'one_validate_text.txt'),
                      os.path.join(dst_dir, 'one_test_text.txt')], 
                      os.path.join(dst_dir, 'vocab_cased.pickle'))

    '''build_dictionary([os.path.join(dst_dir, 'moview_train_text.txt'), 
                      os.path.join(dst_dir, 'moview_validate_text.txt'),
                      os.path.join(dst_dir, 'moview_test_text.txt')], 
                      os.path.join(dst_dir, 'moview_vocab_cased.pickle'))'''

