#coding:utf-8
import tensorflow as tf
import numpy as np
import random
import sys
import csv
from preprocess_ch import load_data
from preprocess_ch import data2vector


class ch_model(object):
    def __init__(self,config):
        self._batch_size = config.batch_size
        self._input_dim = config.input_dim
        self._out_dim = config.num_class
        self._num_class = config.num_class
        self._doc_len = config.doc_len
        self._pf_dim = config.pf_dim
        self._lr = config.lr

        self._new_word_embed = tf.placeholder(tf.float32, shape=[config.vocab_len, config.vocab_dim], name="new_w_embed")

        self._word_embed = tf.get_variable("w_embed", shape=[config.vocab_len, config.vocab_dim], trainable=False)
        self._PF_R1 = tf.get_variable("pf_r1", shape=[config.doc_len*2, config.pf_dim], trainable=False)
        self._PR_R2 = tf.get_variable("pf_r2", shape=[config.doc_len*2, config.pf_dim], trainable=False)
        mat = self.generate_unit_mat()
        self._label_matrix = tf.cast(mat, tf.float32)

        # setting of convolution
        self._stride = config.stride
        self._conv_1 = config.conv_1
        self._pool_1 = config.pool_1

        self._w_1 = tf.get_variable("w_1", shape=self._conv_1, dtype=tf.float32)
        self._b_1 = tf.get_variable("b_1", shape=self._conv_1[3], dtype=tf.float32)
        self._w_s = tf.get_variable("w_s", shape=[self._conv_1[3], self._out_dim], dtype=tf.float32)
        self._b_s = tf.get_variable("b_s", shape=[self._out_dim], dtype=tf.float32)
        self.saver = tf.train.Saver(tf.all_variables())

    def generate_unit_mat(self):
        mat = np.zeros([self._out_dim, self._out_dim])
        for i in range(self._out_dim):
            mat[i][i] = 1
        return mat

    def assign_word_embed(self, session, word_embed):
        embed_update = tf.assign(self._word_embed, self._new_word_embed)
        session.run(embed_update, feed_dict={self._new_word_embed: word_embed})

    def assign_pf_embed(self, session, word_embed):
        embed_update = tf.assign(self._word_embed, self._new_word_embed)
        session.run(embed_update, feed_dict={self._new_word_embed: word_embed})

    def inference(self, doc_datas, pf_r1s, pf_r2s, labels):

        word = tf.nn.embedding_lookup(self._word_embed, doc_datas)
        pf_r1 = tf.nn.embedding_lookup(self._PF_R1, pf_r1s)
        pf_r2 = tf.nn.embedding_lookup(self._PR_R2, pf_r2s)
        labels = tf.nn.embedding_lookup(self._label_matrix, labels)
        labels = tf.reshape(labels, [-1, self._out_dim])
        inputs = tf.concat(2, [word, pf_r1, pf_r2])
        inputs = tf.reshape(inputs, [-1, self._doc_len, self._input_dim, 1])

        out_conv_1 = tf.nn.relu(self.conv2d(inputs, self._w_1) + self._b_1)
        out_pool_1 = self.max_pool_all(out_conv_1, self._pool_1)

        outs = tf.reshape(out_pool_1, [-1, self._conv_1[3]])

        logit = tf.matmul(outs, self._w_s) + self._b_s
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit, labels))
        correct_pre = tf.equal(tf.argmax(logit, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
        return loss, accuracy

    def train(self, loss):
        optimizer = tf.train.AdadeltaOptimizer(self._lr)
        #optimizer = tf.train.GradientDescentOptimizer(1)
        train_op = optimizer.minimize(loss)
        return train_op

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=self._stride, padding="VALID")

    def max_pool_all(self, x, pool_size):
        return tf.nn.max_pool(x, ksize=pool_size, strides=self._stride, padding="VALID")


class Config():
    doc_len = 50
    data_len = 1000
    vocab_len = 100
    vocab_dim = 100
    batch_size = 128
    csv_file = "../out/data_ch.csv"
    model_path = "../out/model/ch_model"
    pf_dim = 5
    input_dim = 100+pf_dim*2
    num_class = 10
    lr = 0.01
    max_epoch = 2

    stride = [1, 1, 1, 1]
    conv_1 = [3, input_dim, 1, 230]
    pool_1 = [1, doc_len-2, 1, 1]
    #pool_1 = [doc_len, ]
    #soft_layer = []


def read_csv(filename_queue, input_dim, doc_len):
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[1]]*input_dim
    data = tf.decode_csv(value, record_defaults)
    doc_data = tf.cast(data[0: doc_len], tf.int32)
    pf_r1 = tf.cast(data[doc_len: doc_len*2], tf.int32)
    pf_r2 = tf.cast(data[doc_len*2: doc_len*3], tf.int32)
    label = tf.cast(data[doc_len*3: input_dim], tf.int32)

    return doc_data, pf_r1, pf_r2, label


def read_batch(filename, config, is_shuffle):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=config.max_epoch)
    doc_data, pf_r1, pf_r2, label = read_csv(filename_queue, config.doc_len*3+1, config.doc_len)
    if is_shuffle:
        doc_datas, pf_r1s, pf_r2s, labels = tf.train.shuffle_batch([doc_data, pf_r1, pf_r2, label],
                                  batch_size=config.batch_size,
                                  num_threads=2,
                                  capacity=3*config.batch_size,
                                  min_after_dequeue=config.batch_size)
    else:
        doc_datas, pf_r1s, pf_r2s, labels = tf.train.batch([doc_data, pf_r1, pf_r2, label],
                                                                   batch_size=config.batch_size,
                                                                   num_threads=2,
                                                                   capacity=3 * config.batch_size)
    return doc_datas, pf_r1s, pf_r2s, labels


def cnn_train(config, data_len, embed, pf_r1, pf_r2):
    config.data_len = data_len
    tf.reset_default_graph()
    with tf.Session() as session:
        # build model
        with tf.variable_scope("cnn_ch", reuse=None):
            m_train = ch_model(config)
        with tf.variable_scope("cnn_ch", reuse=True):
            m_valid = ch_model(config)

        doc_datas, pf_r1s, pf_r2s, labels = read_batch(config.csv_file, config, True)
        doc_datas_v, pf_r1s_V, pf_r2s_v, labels_v = read_batch(config.csv_file, config, False)


        for item in tf.all_variables():
            print "var: ", item
        for item in tf.local_variables():
            print "local:", item

        loss, _ = m_train.inference(doc_datas, pf_r1s, pf_r2s, labels)
        loss_v, acc_v = m_valid.inference(doc_datas_v, pf_r1s_V, pf_r2s_v, labels_v)
        train_op = m_train.train(loss)

        tf.initialize_all_variables().run()
        tf.initialize_local_variables().run()
        m_train.assign_word_embed(session, embed)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=session)
        
        epoch = 0
        step = 0
        min_cost = sys.maxint
        try:
            while not coord.should_stop():
                _, f_l = session.run([train_op, loss])
                step += 1
                if step == config.data_len // config.batch_size:
                    cost = 0.0
                    acc = 0.0
                    for i in range(step):
                        v_l, acc_l = session.run([loss_v, acc_v])
                        cost += v_l
                        acc += acc_l
                    cost /= step
                    acc /= step
                    if cost < min_cost:
                        min_cost = cost
                        print "save model as cost:", cost
                        m_train.saver.save(session, config.model_path)
                    print "epoch: ", epoch, "loss: ", cost, "acc: ", acc, "step:", step
                    step = 0
                    epoch += 1
        except tf.errors.OutOfRangeError:
            print("Done training")
        finally:
            coord.request_stop()
        coord.join(threads)


def cnn_test(config, data_len):
    config.data_len = data_len
    tf.reset_default_graph()
    config.max_epoch = 1
    with tf.Session() as session:
        # build model
        with tf.variable_scope("cnn_ch", reuse=None):
            m_valid = ch_model(config)

        doc_datas_v, pf_r1s_V, pf_r2s_v, labels_v = read_batch(config.csv_file, config, False)


        loss_v, acc_v = m_valid.inference(doc_datas_v, pf_r1s_V, pf_r2s_v, labels_v)
        m_valid.saver.restore(session, config.model_path)

        for item in tf.all_variables():
            print "var:", item
        for item in tf.local_variables():
            print "local:", item
        tf.initialize_local_variables().run()


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=session)
        step = 0
        cost = 0.0
        acc = 0.0
        try:
            while not coord.should_stop():
                v_l, acc_l = session.run([loss_v, acc_v])
                cost += v_l
                acc += acc_l
                step += 1
        except tf.errors.OutOfRangeError:
            cost /= step
            acc /= step
            print "loss: ", cost, "acc: ", acc
            print("Done testing")
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    config = Config()
    config.csv_file = "out/data_ch.csv"
    config.model_path="out/model/ch_model_all"

    raw_data_path = "Raw_Data/ten_relations"
    seg_model_path = "data_tool/seg_tool/cws.model"
    dic_model_path = "data_tool/seg_tool/vocab.txt"
    vocab_path = "data_tool/word2vec/vectors.txt"
    seg_model = load_data.seg_initialize(seg_model_path, dic_model_path)
    vocab, embed, _ = load_data.load_word_embedding(vocab_path)
    data_len, relation = load_data.load_seg_data(raw_data_path, vocab, seg_model, config.csv_file)

    '''
    embed = np.random.random(size=(100, 10))
    f_in = open(config.csv_file, "wb")
    writer = csv.writer(f_in)
    for i in range(100):
        temp = list()
        for k in range(20):
            temp.append(random.randint(i/50, i/50+50))
        for k in range(40):
            temp.append(random.randint(0, 30))
        temp.append(i/50)
        if len(temp) != 61:
            break
        writer.writerow(tuple(temp))

    config.num_class = 2
    config.doc_len = 20
    config.vocab_len = len(embed)
    config.vocab_dim = len(embed[0])
    config.data_len = 100
    config.input_dim = 20
    config.conv_1 = [3, config.input_dim, 1, 230]
    config.pool_1 = [1, config.doc_len - 2, 1, 1]
    '''
    print relation
    config.num_class = len(relation)
    config.vocab_len = len(embed)
    config.max_epoch = 1000
    pf_r1, pf_r2 = data2vector.generate_position_embedding(scale=config.doc_len, dim=config.pf_dim)
    cnn_train(config, data_len, embed, pf_r1, pf_r2)
    #cnn_test(config, data_len)
