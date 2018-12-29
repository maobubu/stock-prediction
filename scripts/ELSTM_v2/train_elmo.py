import numpy
import os

from elstm import train

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
    batch_size       = 8,
    valid_batch_size = 8,
    dispFreq         = 20,
    validFreq        = int(1411/8+1),#1421  #1391
    saveFreq         = int(1411/8+1),
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
    datasets         = ['/local/scratch/yunke_ws/data/data/ding_new_10/train.csv',
                        '/local/scratch/yunke_ws/data/data/ding_new_10/train_label.csv'],
    valid_datasets   = ['/local/scratch/yunke_ws/data/data/ding_new_10/validate.csv',
                        '/local/scratch/yunke_ws/data/data/ding_new_10/validate_label.csv'],
    test_datasets    = ['/local/scratch/yunke_ws/data/data/ding_new_10/test.csv',
                        '/local/scratch/yunke_ws/data/data/ding_new_10/test_label_1.csv'],
    tech_data        = '/local/scratch/yunke_ws/data/data/ding_new_10/technical.csv',
    dictionary       = '/local/scratch/yunke_ws/data/data/ding_new_10/vocab_cased_title.pickle',
    embedding        = '/local/scratch/yunke_ws/data/emb/vectors_latest.txt',
    wait_N           = 1
    )