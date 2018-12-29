import os
from tensorflow.core.protobuf import config_pb2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy
from collections import defaultdict
import tensorflow as tf
import logging
import math
from tensorflow import logging  as log
from tensorflow.python import debug as tf_debug
from collections import OrderedDict
from data_iterator import TextIterator
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers
import warnings
import pickle as pkl
import sys
import pprint
import pdb
import os
import copy
import time
# import matplotlib.pyplot as plt
import pickle
import math
import re
import numpy as np
import random
from tensorflow.python.ops import array_ops

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 300
VOCAB = 220000
LSTM_KEEP_PROB = 0.5
LSTM_DIMENSION = 100
DENSE_KEEP_PROB = 0.8
MAX_NEWS = 80
MAX_WORDS = 1000
BATCH_SIZE = 2
LEARNING_RATE1 = 0.001
GRADIENT_CLIP = 5.
PATIENT = 30
L2_LAMBDA = 0
WORD_EMBEDDING_PATH = '/Users/zhangyunke/Downloads/glove.840B.300d.txt'


def focal_loss(labels, logits, gamma=2.0, alpha=0.25):
    """
    focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: logits is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022
    :param labels: ground truth labels, shape of [batch_size]
    :param logits: model's output, shape of [batch_size, num_cls]
    :param gamma:
    :param alpha:
    :return: shape of [batch_size]
    """
    epsilon = 1.e-9
    labels = tf.to_int64(labels)
    labels = tf.convert_to_tensor(labels, tf.int64)
    logits = tf.convert_to_tensor(logits, tf.float32)
    num_cls = logits.shape[1]

    model_out = tf.add(logits, epsilon)
    onehot_labels = tf.one_hot(labels, num_cls)
    ce = tf.multiply(onehot_labels, -tf.log(model_out))
    weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
    return reduced_fl


def get_data(news_batches):
    max_sequence = max(len(z) for i in news_batches for z in i)  # get max length of news
    if max_sequence > MAX_WORDS:
        max_sequence = MAX_WORDS
    max_news = max(len(i) for i in news_batches)
    #print(max_sequence, max_news)
    padding_batches = numpy.zeros((len(news_batches), max_news, max_sequence)).astype('int64')
    news_mask = numpy.zeros((len(news_batches), max_news)).astype('float32')
    for batch in range(len(news_batches)):
        for news in range(len(news_batches[batch])):
            for words in range(len(news_batches[batch][news])):
                try:
                    padding_batches[batch][news][words] = news_batches[batch][news][words]  # 生成padding以后的文章序列
                except IndexError:
                    continue
    for batch in range(len(news_batches)):
        for news in range(len(news_batches[batch])):
            news_mask[batch][news] = 1.
        #print(len(news_batches[batch]))
    #print(padding_batches)
    #print(news_mask)
    return padding_batches, news_mask


def get_epoch_data(data, data_abstract, shuffle=False):
    data_news = []
    data_abstracts = []
    labels = []
    for _, (x, x_d1, x_d2, y) in enumerate(data):
        x = flatten_news(x)
        data_news.append(x[0])
        labels.append(y[0])
    for _, (x, x_d1, x_d2, y) in enumerate(data_abstract):
        x = flatten_news(x)
        data_abstracts.append(x[0])
    if shuffle is True:
        samples = list(zip(data_news, data_abstracts, labels))
        random.shuffle(samples)
        data_news[:], data_abstracts[:], labels[:] = zip(*samples)
        return data_news, data_abstracts, labels
    else:
        return data_news, data_abstracts, labels


def flatten_news(batches):  # 输入是固定的n个batch的数据
    output = []
    for batch in range(len(batches)):  # j: 1 to 30
        one_batch = []
        # print(len(x[batch])) # 每一天多少篇新闻
        for news in range(len(batches[batch])):
            # print(len(x[batch][news]))  # 每一篇新闻几句话
            # print(x[batch][news])
            # print(np.hstack(x[batch][news]).tolist())  # 把断句过的每一篇新闻拍平
            one_batch.append(np.hstack(batches[batch][news]).tolist())
        output.append(one_batch)
    #print(len(output))
    #print(output)
    return output


def cleansentences(string):
    string = string.replace("<br />", " ")
    return re.sub(re.compile("[^A-Za-z0-9 ]+"), "", string.lower())


def _s(pp, name):  # add perfix
    return '{}_{}'.format(pp, name)


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('{} is not in the archive'.format(kk))
            continue
        params[kk] = pp[kk]

    return params


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * numpy.sqrt(6.0 / (fan_in + fan_out))
    high = constant * numpy.sqrt(6.0 / (fan_in + fan_out))
    W = numpy.random.uniform(low=low, high=high, size=(fan_in, fan_out))
    return W.astype('float32')


def ortho_weight(ndim):  # used by norm_weight below
    """
    Random orthogonal weights
    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        # W = numpy.random.uniform(-0.5,0.5,size=(nin,nout))
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def attention_v1(input, masks, is_training, name='attention', nin=600, keep=1.0, hidden_dim = 300, actvation_f=tf.nn.tanh, origin=None):
    # input is batch,time_step,hidden_state (32*40)*13*600 mask (32*40)*13
    # hidden layer is:batch,hidden_shape,attention_hidden_size (32*40)*13*1200 or (32*40)*13*600
    # attention shape after squeeze is (32*40)*13, # batch,time_step,attention_size (32*40)*13*1
    with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
        hidden = tf.layers.dense(input, hidden_dim, activation=None, use_bias=True,
                                 kernel_initializer=layers.xavier_initializer(uniform=True, seed=None,
                                                                              dtype=tf.float32),
                                 name='hidden', reuse=tf.AUTO_REUSE)
        hidden = tf.layers.batch_normalization(hidden, trainable=True)
        hidden = tf.tanh(hidden, name="activation")
        hidden = layers.dropout(hidden, keep_prob=keep, is_training=is_training)
        # hidden = tf.layers.batch_normalization(hidden, training=is_training)
        # hidden=tf.nn.tanh(hidden)
        attention = tf.layers.dense(hidden, 1, activation=None, use_bias=False,
                                    kernel_initializer=layers.xavier_initializer(uniform=True, seed=None,
                                                                                 dtype=tf.float32),
                                    name='out',
                                    reuse=tf.AUTO_REUSE)
        padding = tf.fill(tf.shape(attention), float('-1e8'))  # float('-inf')
        attention = tf.where(tf.equal(tf.expand_dims(masks, -1), 0.), padding,
                             attention)  # fill 0 with a small number for softmax
        attention = tf.nn.softmax(attention, 1, name='softmax') * tf.expand_dims(masks, -1)  # 32*40*r #mask the attention here is not really neccesary,
        if origin is not None:
            results = tf.reduce_sum(origin * attention, axis=1)
        else:
            results = tf.reduce_sum(input * attention, axis=1)  # 32*600
        '''
        input1 = tf.slice(input, [0, 0, 0], [tf.shape(input)[0], tf.shape(input)[1], 300])
        input2 = tf.slice(input, [0, 0, 300], [tf.shape(input)[0], tf.shape(input)[1], 300])
        #attention1 = tf.slice(attention, [0, 0, 0], [tf.shape(attention)[0], tf.shape(attention)[1], 300])
        #attention2 = tf.slice(attention, [0, 0, 300], [tf.shape(attention)[0], tf.shape(attention)[1], 300])
        result1 = input1 * attention
        result2 = input2 * attention
        results = tf.concat([result1, result2], axis=2)
        results = tf.reduce_sum(results, axis=1)
        '''
    return results


def lstm_filter(input, mask, keep_prob, prefix='lstm', dim=300, is_training=True):
    with tf.variable_scope(name_or_scope=prefix, reuse=tf.AUTO_REUSE):
        sequence = tf.cast(tf.reduce_sum(mask, 1), tf.int32)
        lstm_fw_cell = rnn.LSTMCell(dim, forget_bias=0.0, initializer=tf.orthogonal_initializer(), state_is_tuple=True)
        keep_rate = tf.cond(is_training is not False and keep_prob < 1, lambda: 0.8, lambda: 1.0)
        cell_dp_fw = rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=keep_rate)
        outputs, _ = tf.nn.dynamic_rnn(cell_dp_fw, input, sequence_length=sequence, swap_memory=False,
                                       dtype=tf.float32)
    return outputs


def bilstm_filter(input, mask, keep_prob, is_training, dim=300, prefix='lstm'):
    with tf.variable_scope(name_or_scope=prefix, reuse=tf.AUTO_REUSE):
        sequence = tf.cast(tf.reduce_sum(mask, 1), tf.int32)
        lstm_fw_cell = rnn.LSTMBlockCell(dim,
                                         forget_bias=1.0)  # initializer=tf.orthogonal_initializer(), state_is_tuple=True
        # back directions
        lstm_bw_cell = rnn.LSTMBlockCell(dim, forget_bias=1.0)
        keep_rate = tf.cond(is_training, lambda: 0.8, lambda: 1.0)
        cell_dp_fw = rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=keep_rate)
        cell_dp_bw = rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=keep_rate)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_dp_fw, cell_dp_bw, input, sequence_length=sequence,
                                                     swap_memory=False,
                                                     dtype=tf.float32)  # batch major
    return outputs


def init_params(options, worddicts):
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(220000, 300)
    # read embedding from GloVe
    if options['embedding']:
        with open(WORD_EMBEDDING_PATH, 'r') as f:
            for line in f:
                tmp = line.split()
                '''
                x = len(tmp) - 300
                word = ""
                if x > 0:
                    words = tmp[0:x]
                    word_ = " "
                    word = word_.join(words)
                else:
                    word = tmp[0].lower()
                vector = tmp[x:]
                '''
                if len(tmp) > 301:
                    continue
                else:
                    word = cleansentences(tmp[0].lower())
                    vector = tmp[1:]

                # print(len(vector))
                if word in worddicts and worddicts[word] < 220000:
                    try:
                        params['Wemb'][worddicts[word], :] = vector
                        # encoder: bidirectional RNN
                    except ValueError as e:
                        print(str(e))
                        print(len(vector))
    return params


def word_embedding(options, params):
    print("word embedding")
    embeddings = tf.get_variable("embeddings", shape=[220000, 300],
                                 initializer=tf.constant_initializer(numpy.array(params['Wemb'])))
    return embeddings


def build_model(embedding):
    print("Start building models!")
    with tf.device('/cpu:0'):
        with tf.variable_scope('input'):
            '''
            定义图的初始三个输入， x:新闻矩阵 y:标签矩阵 z:摘要矩阵， 其中新闻矩阵和摘要矩阵尚未embedding
            由于不存在句子维度，故取消了sentence mask
            '''
            # DEFINE INPUT OF THE GRAPH
            epoch = tf.placeholder(tf.float32, shape=None, name="epoch")
            x = tf.placeholder(tf.int64, shape=[None, None, None], name='x')
            # batch * news * sequence
            x_sequence_mask = tf.cast(tf.abs(tf.sign(x)), tf.float32, name="x_sequence_mask")
            # batch * news * sequence(mask)
            x_news_mask = tf.placeholder(tf.float32, shape=[None, None], name='x_news_mask')  # mask news, batch
            # batch * news
            y = tf.placeholder(tf.int64, shape=[None], name='y')
            # label
            z = tf.placeholder(tf.int64, shape=[None, None, None], name='z')
            # batch * abtsract * sequence
            z_sequence_mask = tf.cast(tf.abs(tf.sign(z)), tf.float32, name="z_sequence_mask")
            # batch * abstract * sequence(mask)
            z_abstract_mask = tf.placeholder(tf.float32, shape=[None, None], name='z_news_mask')  # mask abstract, batch
            # batch * abstract

            keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
            is_training = tf.placeholder(tf.bool, name='is_training')

            # WORD EMBEDDING OF X AND Z
            news_emb = tf.nn.embedding_lookup(embedding, x)
            # Batch * news * sequence * embedding
            abstract_emb = tf.nn.embedding_lookup(embedding, z)
            # Batch * abstracts * sequence * embdding

    with tf.device('/gpu:0'):
        with tf.variable_scope('abstract'):
            '''
            定义abstract计算图：abstract -> bilstm -> self_attention -> abstract_attention
            '''
            batch = tf.shape(abstract_emb)[0]
            abstract = tf.shape(abstract_emb)[1]
            #news = abstract
            news = tf.shape(news_emb)[1]
            abstract_word = tf.shape(abstract_emb)[2]
            news_word = tf.shape(news_emb)[2]
            # notice that batch, news number of abstract and news are the same
            # only differene is words

            abstract_input = tf.reshape(abstract_emb, [batch * abstract, abstract_word, EMBEDDING_DIM])
            # (Batch * abstracts) * sequence * embedding
            abstract_sequence_mask = tf.reshape(z_sequence_mask, [batch * abstract, abstract_word])
            # (Batch * abstracts) * sequence
            abstract_encode = bilstm_filter(abstract_input, abstract_sequence_mask, LSTM_KEEP_PROB, is_training, dim=LSTM_DIMENSION)
            # output dimension is Batch * abstracts * sequence * (LSTM_DIMENSION * 2)
            abstract_encode = tf.concat(abstract_encode, 2) * tf.expand_dims(abstract_sequence_mask, -1)
            # TODO: check if this is necessary: mask the output
            abstract_attention = attention_v1(abstract_encode, abstract_sequence_mask, is_training, name='abstract_attention',
                                              keep=DENSE_KEEP_PROB, hidden_dim=LSTM_DIMENSION)
            # Self attention output is (Batch * abstracts) * 600
            # AKA every abstract is emcoded, and this is use for news attention

    with tf.device('/gpu:1'):
        with tf.variable_scope('news'):
            '''
            定义news计算图：news -> bilstm -> abstract_attention -> self_attention -> mapping -> calculate_loss
            '''
            sequence_input = tf.reshape(news_emb, [batch * news, news_word, EMBEDDING_DIM])
            # (Batch * news) * sequence * embedding
            news_sequence_mask = tf.reshape(x_sequence_mask, [batch * news, news_word])
            # (Batch * news) * sequence
            sequence_encode = bilstm_filter(sequence_input, news_sequence_mask, LSTM_KEEP_PROB, is_training, dim=LSTM_DIMENSION)
            # output dimension is Batch * news * sequence * (LSTM_DIMENSION * 2)
            sequence_encode = tf.concat(sequence_encode, 2) * tf.expand_dims(news_sequence_mask, -1)
            # TODO: check if this is necessary: mask the output
            abstract_attention = tf.reshape(abstract_attention, [batch * abstract, LSTM_DIMENSION * 2])
            sequence_encode = tf.reshape(sequence_encode, [batch * news, news_word, LSTM_DIMENSION * 2])
            # Reshape the prepared tensor
            abstract_attention = tf.expand_dims(abstract_attention, axis=1)
            '''
            This is multiply attention
            news_attention = tf.matmul(abstract_attention, sequence_encode, transpose_b=[0, 2, 1])
            #news_attention = tf.nn.softmax(news_attention, axis=1) * tf.expand_dims(news_sequence_mask, -1)
            news_attention = tf.nn.softmax(tf.squeeze(news_attention), axis=-1)
            news_attention = news_attention * news_sequence_mask
            # (Batch * news) * sequence
            #news_attention = tf.reduce_mean(news_attention, axis=2)
            sequence_output = sequence_encode * tf.expand_dims(news_attention, -1)
            news_encode = tf.reduce_sum(sequence_output, axis=1)
            # (Batch * news) * 600
            news_encode = tf.reshape(news_encode, [batch, news, LSTM_DIMENSION * 2])
            news_output = attention_v1(news_encode, x_news_mask, is_training, name='news_attention',
                                              keep=DENSE_KEEP_PROB)
            '''
            abstract_attention = tf.tile(abstract_attention, [1, news_word, 1])
            sentence_output = tf.concat([abstract_attention, sequence_encode], 2)
            news_encode = attention_v1(sentence_output, news_sequence_mask, is_training, name="sequence_attention",
                                       keep=DENSE_KEEP_PROB, hidden_dim=LSTM_DIMENSION * 2, actvation_f=tf.nn.tanh, origin=sequence_encode)
            news_encode = tf.reshape(news_encode, [batch, news, LSTM_DIMENSION * 2])
            news_output = attention_v1(news_encode, x_news_mask, is_training, name='news_attention',
                                       hidden_dim=LSTM_DIMENSION, keep=DENSE_KEEP_PROB)
            # Duplicate attention encode for contact it on the sentence encode

            # Batch * 600

        with tf.name_scope('final'):

            logit = tf.layers.dense(news_output, LSTM_DIMENSION, activation=tf.nn.relu, use_bias=True,
                                        kernel_initializer=layers.xavier_initializer(uniform=True, seed=None,
                                                                                     dtype=tf.float32),
                                        name='mapping', reuse=tf.AUTO_REUSE)
            # TODO: check if needs mapping dropout
            logit = layers.dropout(logit, keep_prob=DENSE_KEEP_PROB, is_training=is_training)
            pred = tf.layers.dense(logit, 2, activation=None, use_bias=True,
                                   kernel_initializer=layers.xavier_initializer(uniform=True, seed=None,
                                                                                dtype=tf.float32),
                                   name='mapping_out', reuse=tf.AUTO_REUSE)
            logger.info('Building f_cost...')
            '''
            labels = tf.one_hot(y, depth=2, axis=1)
            preds = tf.nn.softmax(pred, 1, name='softmax')
            cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels
            '''
            preds = tf.nn.softmax(pred, 1, name='softmax')
            cost = focal_loss(y, preds)
        logger.info('Done')

        with tf.variable_scope('logging'):
            tf.summary.scalar('current_cost', tf.reduce_mean(cost))
            tf.summary.histogram('predicted_value', preds)
            summary = tf.summary.merge_all()
    return is_training, cost, x, x_news_mask, y, preds, summary, epoch


def predict_pro_acc(sess, cost, model_options, iterator, abstract_iterator, maxlen, correct_pred, pred, summary, eidx,
                    is_training, plot=None, writer=None, validate=False):
    # fo = open(_s(prefix,'pre.txt'), "w")
    print("Start validate and test")
    num = 0
    valid_acc = 0
    total_cost = 0
    loss = 0
    result = 0
    # sess.add_tensor_filter("val_test_spot")
    counter = 0
    iterators, iterator_abstracts, iterator_labels = get_epoch_data(iterator, abstract_iterator, shuffle=False)
    while counter + BATCH_SIZE < len(iterators):
        num += BATCH_SIZE
        batch_news = []
        batch_abstract = []
        batch_label = []
        keep_prob = model_options['keep_prob']
        is_training = True
        for sample in range(BATCH_SIZE):
            batch_news.append(iterators[counter + sample])
            batch_abstract.append(iterator_abstracts[counter + sample])
            batch_label.append(iterator_labels[counter + sample])
        batch_news, batch_news_mask = get_data(batch_news)
        batch_abstract, batch_abstract_mask = get_data(batch_abstract)
        counter += BATCH_SIZE

        loss, result, preds = sess.run([cost, correct_pred, pred],
                                       feed_dict={'input/x:0': batch_news,
                                                  'input/x_news_mask:0': batch_news_mask,
                                                  'input/y:0': batch_label,
                                                  'input/z:0': batch_abstract,
                                                  'input/z_news_mask:0': batch_abstract_mask,
                                                  'input/keep_prob:0': keep_prob,
                                                  'input/is_training:0': True,
                                                  'input/epoch:0': eidx},
                                       options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
        valid_acc += result.sum()
        total_cost += loss.sum()
        if plot is not None:
            if validate is True:
                plot['validate'].append(loss.sum() / BATCH_SIZE)
            else:
                plot['testing'].append(loss.sum() / BATCH_SIZE)
    final_acc = 1.0 * valid_acc / num
    final_loss = 1.0 * total_cost / num
    # if writer is not None:
    #    writer.add_summary(test_summary, eidx)

    # print result,preds,loss,result_
    print(preds, result, num)

    return final_acc, final_loss


def train(
        dim_word=100,  # word vector dimensionality
        dim=100,  # the number of GRU units
        encoder='lstm',  # encoder model
        decoder='lstm',  # decoder model
        patience=10,  # early stopping patience
        max_epochs=500,
        finish_after=10000000,  # finish after this many updates
        decay_c=0.,  # L2 regularization penalty
        clip_c=-1.,  # gradient clipping threshold
        lrate=0.0004,  # learning rate
        n_words=100000,  # vocabulary size
        n_words_lemma=100000,
        maxlen=100,  # maximum length of the description
        optimizer='adam',
        batch_size=32,
        valid_batch_size=32,
        save_model='../../models/',
        saveto='model.npz',
        dispFreq=100,
        validFreq=1000,
        saveFreq=1000,  # save the parameters after every saveFreq updates
        use_dropout=False,
        reload_=False,
        verbose=False,  # print verbose information for debug but slow speed
        delay1=3,
        delay2=7,
        types='title',
        cut_word=False,
        cut_sentence=False,
        cut_news=False,
        last_layer="LSTM",
        CNN_filter=64,
        CNN_kernel=3,
        keep_prob=0.8,
        datasets=[],
        valid_datasets=[],
        test_datasets=[],
        dictionary=[],
        kb_dicts=[],
        embedding='',  # pretrain embedding file, such as word2vec, GLOVE
        dim_kb=5,
        RUN_NAME="histogram_visualization",
        wait_N=10,
        LEARNING_RATE=LEARNING_RATE1
):
    #print("start!!!!!!!!!!!!!!")
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s",
                        filename='./log_result.txt')
    # Model options
    model_options = locals().copy()
    #print(model_options)
    # tf.set_random_seed(2345)
    with open(dictionary, 'rb') as f:
        worddicts = pkl.load(f)

    logger.info("Loading knowledge base ...")

    # reload options
    if reload_ and os.path.exists(saveto):
        logger.info("Reload options")
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    logger.debug(pprint.pformat(model_options))

    logger.info("Loading data")
    #print(batch_size, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #cut_word = 50
    #cut_sentence = 50
    print("Reading news from the original file!")
    train = TextIterator(datasets[0], datasets[1],
                         dict=dictionary,
                         delay1=delay1,
                         delay2=delay2,
                         types="abstract",
                         n_words=VOCAB,
                         batch_size=1,
                         cut_word=False,
                         cut_sentence=False,
                         cut_news=MAX_NEWS,
                         shuffle=False, shuffle_sentence=False)

    train_abstract = TextIterator(datasets[0], datasets[1],
                         dict=dictionary,
                         delay1=delay1,
                         delay2=delay2,
                         types="title",
                         n_words=VOCAB,
                         batch_size=1,
                         cut_word=False,
                         cut_sentence=False,
                         cut_news=MAX_NEWS,
                         shuffle=False, shuffle_sentence=False)

    valid = TextIterator(valid_datasets[0], valid_datasets[1],
                         dict=dictionary,
                         delay1=delay1,
                         delay2=delay2,
                         types="abstract",
                         n_words=VOCAB,
                         batch_size=1,
                         cut_word=False,
                         cut_sentence=False,
                         cut_news=MAX_NEWS,
                         shuffle=False, shuffle_sentence=False)

    valid_abstract = TextIterator(valid_datasets[0], valid_datasets[1],
                         dict=dictionary,
                         delay1=delay1,
                         delay2=delay2,
                         types="title",
                         n_words=VOCAB,
                         batch_size=1,
                         cut_word=False,
                         cut_sentence=False,
                         cut_news=MAX_NEWS,
                         shuffle=False, shuffle_sentence=False)

    test = TextIterator(test_datasets[0], test_datasets[1],
                        dict=dictionary,
                        delay1=delay1,
                        delay2=delay2,
                        types="abstract",
                        n_words=VOCAB,
                        batch_size=1,
                        cut_word=False,
                        cut_sentence=False,
                        cut_news=MAX_NEWS,
                        shuffle=False, shuffle_sentence=False)

    test_abstract = TextIterator(test_datasets[0], test_datasets[1],
                        dict=dictionary,
                        delay1=delay1,
                        delay2=delay2,
                        types="title",
                        n_words=VOCAB,
                        batch_size=1,
                        cut_word=False,
                        cut_sentence=False,
                        cut_news=MAX_NEWS,
                        shuffle=False, shuffle_sentence=False)

    # Initialize (or reload) the parameters using 'model_options'
    # then build the tensorflow graph
    logger.info("init_word_embedding")
    params = init_params(model_options, worddicts)
    embedding = word_embedding(model_options, params)
    is_training, cost, x, x_mask, y, pred, summary, epoch = build_model(embedding)
    with tf.variable_scope('train'):
        lr = tf.Variable(0.0, trainable=False)

        def assign_lr(session, lr_value):
            session.run(tf.assign(lr, lr_value))

        logger.info('Building optimizers...')
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        #optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=0.9)
        logger.info('Done')
        # print all variables
        tvars = tf.trainable_variables()
        for var in tvars:
            print(var.name, var.shape)
        L2_regu = L2_LAMBDA * tf.add_n([tf.nn.l2_loss(v) for v in tvars if ('embeddings' not in v.name and 'bias' not in v.name)])
        cost = cost + L2_regu
        square = 0
        for t in tf.gradients(cost, tvars):
            square = tf.add(tf.square(tf.reduce_mean(t)), square)
        global_norm = tf.sqrt(square)
        # global_norm = tf.sqrt(tf.reduce_sum(tf.square([ for t in tf.gradients(cost, tvars)])))
        logging.debug("global gradient sum square is " + str(global_norm))
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), GRADIENT_CLIP)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # 定义滑动平均
        # moving_averages = tf.train.ExponentialMovingAverage(0.99, epoch)
        # moving_averages_op = moving_averages.apply(tvars)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.apply_gradients(zip(grads, tvars))
            # moving_averages_op = moving_averages.apply(tvars)
            # train_op = tf.no_op
        # train_op = optimizer.minimize(cost)
        op_loss = tf.reduce_mean(cost)
        logger.info("correct_pred")
        correct_pred = tf.equal(tf.argmax(input=pred, axis=1), y)  # make prediction
    logger.info("Done")

    temp_accuracy = tf.cast(correct_pred, tf.float32)  # change to float32

    logger.info("init variables")
    init = tf.global_variables_initializer()
    logger.info("Done")
    print("Finished!")
    # saver
    saver = tf.train.Saver(max_to_keep=15)

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        training_writer = tf.summary.FileWriter("./logs/{}/training".format(RUN_NAME), sess.graph)
        validate_writer = tf.summary.FileWriter("./logs/{}/validate".format(RUN_NAME), sess.graph)
        testing_writer = tf.summary.FileWriter("./logs/{}/testing".format(RUN_NAME), sess.graph)
        sess.run(init)
        history_errs = []
        # reload history
        if reload_ and os.path.exists(saveto):
            logger.info("Reload history error")
            history_errs = list(numpy.load(saveto)['history_errs'])

        bad_counter = 0

        if validFreq == -1:
            validFreq = len(train[0]) / BATCH_SIZE
        if saveFreq == -1:
            saveFreq = len(train[0]) / BATCH_SIZE

        loss_plot = defaultdict(list)
        uidx = 0
        estop = False
        valid_acc_record = []
        test_acc_record = []
        best_test_accuracy = 0
        best_valid_cost = 100
        best_num = -1
        best_valid_num = 0
        best_test_num = 0
        valid_counter = 0
        lr_change_list = []
        #learning_rate = LEARNING_RATE
        early_stop_counter = 0
        optimizer_flag = 0
        print("Strat extracting news and training!")
        for eidx in range(max_epochs):
            assign_lr(sess, LEARNING_RATE)
            n_samples = 0
            training_cost = 0
            training_acc = 0
            train_news, train_abstracts, train_labels = get_epoch_data(train, train_abstract, shuffle=True)
            data_counter = 0
            while data_counter + BATCH_SIZE < len(train_news):
                batch_news = []
                batch_abstract = []
                batch_label = []
                n_samples += BATCH_SIZE
                uidx += 1
                keep_prob = LSTM_KEEP_PROB
                is_training = True
                for sample in range(BATCH_SIZE):
                    batch_news.append(train_news[data_counter + sample])
                    batch_abstract.append(train_abstracts[data_counter + sample])
                    batch_label.append(train_labels[data_counter + sample])
                batch_news, batch_news_mask = get_data(batch_news)
                batch_abstract, batch_abstract_mask = get_data(batch_abstract)
                data_counter += BATCH_SIZE
                #print(batch_news.shape, batch_abstract.shape)
                if batch_news.shape[1] != batch_abstract.shape[1]:
                    continue
                if batch_news is None:
                    logger.debug('Minibatch with zero sample under length {0}'.format(maxlen))
                    uidx -= 1
                    continue
                ud_start = time.time()
                _, loss, loss_no_mean, temp_acc = sess.run([train_op, op_loss, cost, temp_accuracy],
                                                                     feed_dict={'input/x:0': batch_news,
                                                                                'input/x_news_mask:0': batch_news_mask,
                                                                                'input/y:0': batch_label,
                                                                                'input/z:0': batch_abstract,
                                                                                'input/z_news_mask:0': batch_abstract_mask,
                                                                                'input/keep_prob:0': keep_prob,
                                                                                'input/is_training:0': True,
                                                                                'input/epoch:0': eidx})
                print(batch_news.shape, batch_abstract.shape)
                ud = time.time() - ud_start
                training_cost += loss_no_mean.sum()
                training_acc += temp_acc.sum()
                loss_plot['training'].append(loss)
                '''train_summary = sess.run(summary, feed_dict={'input/x:0': data_x, 'input/x_mask:0': data_x_mask,
                                                              'input/y:0': data_y,'input/keep_prob:0':keep_prob,'input/is_training:0':is_training})
                training_writer.add_summary(train_summary, eidx)'''
                if numpy.mod(uidx, dispFreq) == 0:
                    logger.debug('Epoch {0} Update {1} Cost {2} TIME {3}'.format(eidx, uidx, loss, ud))

                # validate model on validation set and early stop if necessary
                #if numpy.mod(uidx, validFreq) == 0:
            is_training = False

            valid_acc, valid_loss = predict_pro_acc(sess, cost, model_options, valid,
                                                    valid_abstract, maxlen, correct_pred, pred, summary, eidx,
                                                    is_training, loss_plot, validate_writer, validate=True)
            test_acc, test_loss = predict_pro_acc(sess, cost, model_options, test, test_abstract, maxlen,
                                                  correct_pred, pred, summary, eidx, is_training, loss_plot,
                                                  testing_writer)
            # valid_err = 1.0 - valid_acc
            valid_err = valid_loss
            history_errs.append(valid_err)
            loss_plot['validate_ep'].append(valid_loss)
            loss_plot['testing_ep'].append(test_loss)
            logger.debug('Epoch  {0}'.format(eidx))
            logger.debug('Valid cost  {0}'.format(valid_loss))
            logger.debug('Valid accuracy  {0}'.format(valid_acc))
            logger.debug('Test cost  {0}'.format(test_loss))
            logger.debug('Test accuracy  {0}'.format(test_acc))
            logger.debug('learning_rate:  {0}'.format(LEARNING_RATE))

            valid_acc_record.append(valid_acc)
            test_acc_record.append(test_acc)
            '''
            if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                best_num = best_num + 1
                best_epoch_num = eidx
                wait_counter = 0
                logger.info("Saving...")
                saver.save(sess, _s(_s(_s(save_model, "epoch"), str(best_num)), "model.ckpt"))
                logger.info(_s(_s(_s(save_model, "epoch"), str(best_num)), "model.ckpt"))
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('{}.pkl'.format(saveto), 'wb'))
                logger.info("Done")
            '''

            if valid_loss < best_valid_cost:
                early_stop_counter = 0
                valid_counter = 0
                logger.info("Saving...")
                logger.debug('Valid cost  {0}'.format(valid_loss))
                saver.save(sess, _s(_s(_s(save_model, "epoch"), str(eidx)), "model.ckpt"))
                logger.info(_s(_s(_s(save_model, "epoch"), str(eidx)), "model.ckpt"))
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('{}.pkl'.format(saveto), 'wb'))
                best_valid_cost = valid_loss
                best_epoch_num = eidx
                logger.info("New best valid loss is " + str(valid_loss))
                logger.info("Done")
            if test_acc > best_test_accuracy:
                best_test_accuracy = test_acc
                best_test_num = eidx
            #logger.debug('Test accuracy is HERE!!!!!!!!!!!!!!!')
            logger.debug('So far best test acc is {0} and its at {1} epoch'.format(best_test_accuracy, best_test_num))
            logger.debug('So far best valid loss is {0} and its at {1} epoch'.format(best_valid_cost, best_epoch_num))

            if valid_loss > best_valid_cost:
                logger.debug("Validation loss increase, loading previous model and half learning rate!")
                saver.restore(sess, _s(_s(_s(save_model, "epoch"), str(best_epoch_num)), "model.ckpt"))
                logger.debug("Restore " + str(best_epoch_num) + " model!")
                early_stop_counter += 1
                logger.debug("Early stop num is " + str(early_stop_counter))
                LEARNING_RATE *= 0.8
                if LEARNING_RATE < 0.000001 and optimizer_flag == 0:
                    optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=0.9)
                    optimizer_flag = 1

            '''
            if valid_loss > best_valid_cost:
                early_stop_counter += 1
                LEARNING_RATE = LEARNING_RATE * 0.5
                    #print("Finish one batch!")
                    # if valid_err > numpy.array(history_errs).min():
                    # wait_counter += 1
                    # wait_counter +=1 if valid_err>numpy.array(history_errs).min() else 0
            '''

            '''
            if wait_counter >= wait_N:
                logger.info("wait_counter max, need to half the lr")
                # print 'wait_counter max, need to half the lr'
                bad_counter += 1
                wait_counter = 0
                logger.debug('bad_counter:  {0}'.format(bad_counter))
                # TODO change the learining rate
                #learning_rate = learning_rate * 0.5
                # learning_rate = learning_rate
                assign_lr(sess, learning_rate)
                lr_change_list.append(eidx)
                logger.debug('lrate change to:   {0}'.format(learning_rate))
                # print 'lrate change to: ' + str(lrate)

            if bad_counter > patience:
                logger.info("Early Stop!")
                estop = True
                break

            if numpy.isnan(valid_err):
                pdb.set_trace()
            '''
            # finish after this many updates

            if early_stop_counter == PATIENT:
                logger.info("Finished with " + str(best_valid_num) + " epoch model with minimum valid cost"
                            + str(best_valid_cost))
                break

            logger.debug('Seen samples:  {0}'.format(n_samples))
            logger.debug('Training accuracy:  {0}'.format(1.0 * training_acc / n_samples))
            loss_plot['training_ep'].append(training_cost / n_samples)
            # print 'Seen %d samples' % n_samples
            logger.debug('Saved loss_plot pickle')
            with open("important_plot.pickle", 'wb') as handle:
                pkl.dump(loss_plot, handle, protocol=pkl.HIGHEST_PROTOCOL)
            if estop:
                break

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        # Restore variables from disk.
        saver.restore(sess, _s(_s(_s(save_model, "epoch"), str(best_num)), "model.ckpt"))
        keep_prob = 1
        is_training = False
        logger.info('=' * 80)
        logger.info('Final Result')
        logger.info('=' * 80)
        logger.debug('best epoch   {0}'.format(best_epoch_num))

        valid_acc, valid_cost = predict_pro_acc(sess, cost, model_options, valid,
                                                maxlen, correct_pred, pred, summary, eidx, is_training, None)
        logger.debug('Valid cost   {0}'.format(valid_cost))
        logger.debug('Valid accuracy   {0}'.format(valid_acc))
        # print 'Valid cost', valid_cost
        # print 'Valid accuracy', valid_acc

        test_acc, test_cost = predict_pro_acc(sess, cost, model_options, test,
                                              maxlen, correct_pred, pred, summary, eidx, is_training, None)
        logger.debug('Test cost   {0}'.format(test_cost))
        logger.debug('Test accuracy   {0}'.format(test_acc))

        # print 'best epoch ', best_epoch_num
        #train_acc, train_cost = predict_pro_acc(sess, cost, prepare_data, model_options, train_valid,
                                                #maxlen, correct_pred, pred, summary, eidx, is_training, None)
        #logger.debug('Train cost   {0}'.format(train_cost))
        #logger.debug('Train accuracy   {0}'.format(train_acc))
        # print 'Train cost', train_cost
        # print 'Train accuracy', train_acc

        # print 'Test cost   ', test_cost
        # print 'Test accuracy   ', test_acc

        return None


if __name__ == '__main__':
    pass
