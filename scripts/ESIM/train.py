import numpy
import os

from main import train

if __name__ == '__main__':
    model_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    train(
    saveto           = './{}.npz'.format(model_name),
    reload_          = True,
    dim_word         = 100, # or 300 glove 
    dim              = 300,
    patience         = 20,
    n_words          = 40325,  #s p500 46174 #ding 33976 40325
    clip_c           = 10.,
    lrate            = 0.0004,
    optimizer        = 'adam', 
    maxlen           = 120,
    batch_size       = 256,
    valid_batch_size = 256,
    dispFreq         = 1,
    validFreq        = int(1426/256+1),#s p500 1321 # ding 1426
    saveFreq         = int(1426/256+1),
    use_dropout      = True,
    verbose          = False,
    datasets         = ['../../data/ding_origin/one_train_text.txt',
                        '../../data/ding_origin/one_train_label.txt'],
    valid_datasets   = ['../../data/ding_origin/one_validate_text.txt', 
                        '../../data/ding_origin/one_validate_label.txt'],
    test_datasets    = ['../../data/ding_origin/one_test_text.txt', 
                        '../../data/ding_origin/one_test_label.txt'],
    dictionary       = '../../data/ding_origin/vocab_cased.pickle',
    embedding        = '../../data/glove/vectors_all.txt',
    wait_N           = 10,
    )





