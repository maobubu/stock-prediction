import numpy
import os

from slstm import train

if __name__ == '__main__':
    model_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    train(
    saveto           = './{}.npz'.format(model_name),
    reload_          = False,
    dim_word         = 100, # or 300 glove
    dim              = 100,
    patience         = 30,
    n_words          = 22671,  #s p500 46174 #ding 33976 40325
    clip_c           = 10.,
    lrate            = 0.04,
    optimizer        = 'adam',
    maxlen           = None,
    batch_size       = 16,
    valid_batch_size = 16,
    dispFreq         = 20,
    validFreq        = int(1411/16+1),#1421  #1391
    saveFreq         = int(1411/16+1),
    use_dropout      = True,
    verbose          = False,
    delay1           = 3,
    delay2           = 7,
    delay_tech       = 5,
    types            = 'title',
    cut_word         = False,
    cut_news         = 70,
    last_layer       = 'LSTM',
    CNN_filter       = 128,
    CNN_kernel       = 3,
    keep_prob        = 0.5,
    datasets         = ['/Users/zhangyunke/Desktop/upload/ding_new_10/train.csv',
                        '/Users/zhangyunke/Desktop/upload/ding_new_10/train_label.csv'],
    valid_datasets   = ['/Users/zhangyunke/Desktop/upload/ding_new_10/validate.csv',
                        '/Users/zhangyunke/Desktop/upload/ding_new_10/validate_label.csv'],
    test_datasets    = ['/Users/zhangyunke/Desktop/upload/ding_new_10/test.csv',
                        '/Users/zhangyunke/Desktop/upload/ding_new_10/test_label_1.csv'],
    tech_data        = '/Users/zhangyunke/Desktop/upload/ding_new_10/technical.csv',
    dictionary       = '/Users/zhangyunke/Desktop/upload/ding_new_10/vocab_cased_title.pickle',
    embedding        = '/Users/zhangyunke/Desktop/ding_new_4/vectors_latest.txt',
    wait_N           = 1,
    train_elmo       = '/Users/zhangyunke/Desktop/elmo_data/train_elmo_embeddings.hdf5',
    validate_elmo    = '/Users/zhangyunke/Desktop/elmo_data/validate_elmo_embeddings.hdf5',
    test_elmo        = '/Users/zhangyunke/Desktop/elmo_data/test_elmo_embeddings.hdf5'
    )