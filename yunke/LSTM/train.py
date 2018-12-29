import numpy
import os

from main_new import train

if __name__ == '__main__':
    model_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    train(
    saveto           = './{}.npz'.format(model_name),
    reload_          = True,
    dim_word         = 100, # or 300 glove
    dim              = 300,
    patience         = 20,
    n_words          = 22485,  #s p500 46174 #ding 33976 40325
    clip_c           = 10.,
    lrate            = 0.0001,
    optimizer        = 'adam',
    maxlen           = None,
    batch_size       = 16,
    valid_batch_size = 16,
    dispFreq         = 20,
    validFreq        = int(1411/16+1),#1411  #1379
    saveFreq         = int(1411/16+1),
    use_dropout      = True,
    verbose          = False,
    delay1           = 3,
    delay2           = 7,
    types            = 'title',
    cut_word         = False,
    cut_news         = 30,
    last_layer       = 'LSTM',
    CNN_filter       = 128,
    CNN_kernel       = 3,
    keep_prob        = 0.5,
    datasets         = ['../data/ding_new_1/train.csv',
                        '../data/ding_new_1/train_label.csv'],
    valid_datasets   = ['../data/ding_new_1/validate.csv',
                        '../data/ding_new_1/validate_label.csv'],
    test_datasets    = ['../data/ding_new_1/test.csv',
                        '../data/ding_new_1/test_label.csv'],
    dictionary       = '../data/ding_new_1/vocab_cased_title.pickle',
    embedding        = '../emb/vectors_latest.txt',
    wait_N           = 1,
    )





