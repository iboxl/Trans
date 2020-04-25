import tensorflow as tf
import time
import argparse
import random
import numpy as np
import os.path
import math
import timeit
from multiprocessing import JoinableQueue, Queue, Process
from collections import defaultdict


class OldPModel(object):
    def training_data_batch(self, batch_size):
        n_triple = len(self.triple_train)
        rand_idx = np.random.permutation(n_triple)  # 生成随机一个序列，用于打乱训练组
        start = 0
        while start < n_triple:
            start_t = timeit.default_timer()
            end = min(start + batch_size, n_triple)
            # size = end - start
            train_triple_positive = np.asarray([self.triple_train[x] for x in rand_idx[start:end]])
            train_triple_negative = []
            for t in train_triple_positive:
                replace_entity_id = np.random.randint(self.num_entity)
                random_num = np.random.random()

                if self.negative_sampling == 'unif':
                    replace_head_probability = 0.5
                elif self.negative_sampling == 'bern':
                    replace_head_probability = self.relation_property[t[1]]
                else:
                    raise NotImplementedError("Dose not support %s negative_sampling", self.negative_sampling)

                if random_num < replace_head_probability:
                    train_triple_negative.append((replace_entity_id, t[1], t[2]))
                else:
                    train_triple_negative.append((t[0], t[1], replace_entity_id))

            start = end
            prepare_t = timeit.default_timer() - start_t

            yield train_triple_positive, train_triple_negative, prepare_t

    def load_data(self):
        print('loading entity2id.txt ...')
        with open(os.path.join(self.data_dir, 'entity2id.txt')) as f:
            self.entity2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
            self.id2entity = {value: key for key, value in self.entity2id.items()}

        with open(os.path.join(self.data_dir, 'relation2id.txt')) as f:
            self.relation2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in
                                  f.readlines()}
            self.id2relation = {value: key for key, value in self.relation2id.items()}

        def load_triple(self, triplefile):
            triple_list = []  # [(head_id, relation_id, tail_id),...]
            with open(os.path.join(self.data_dir, triplefile)) as f:
                for line in f.readlines():
                    line_list = line.strip().split('\t')
                    assert len(line_list) == 3
                    headid = self.entity2id[line_list[0]]
                    relationid = self.relation2id[line_list[2]]
                    tailid = self.entity2id[line_list[1]]

                    triple_list.append((headid, relationid, tailid))  # h r t

                    self.hr_t[(headid, relationid)].add(tailid)
                    self.tr_h[(tailid, relationid)].add(headid)
            return triple_list

        self.triple_train = load_triple(self, 'train.txt')
        self.triple_test = load_triple(self, 'test.txt')
        self.triple_valid = load_triple(self, 'valid.txt')
        self.triple = np.concatenate([self.triple_train, self.triple_test, self.triple_valid], axis=0)  # triple 拼接

        self.num_relation = len(self.relation2id)
        self.num_entity = len(self.entity2id)
        self.num_triple_train = len(self.triple_train)
        self.num_triple_test = len(self.triple_test)
        self.num_triple_valid = len(self.triple_valid)

        print('entity number: ' + str(self.num_entity))
        print('relation number: ' + str(self.num_relation))
        print('training triple number: ' + str(self.num_triple_train))
        print('testing triple number: ' + str(self.num_triple_test))
        print('valid triple number: ' + str(self.num_triple_valid))

        if self.negative_sampling == 'bern':
            self.relation_property_head = {x: [] for x in
                                             range(self.num_relation)}  # {relation_id:[headid1, headid2,...]}
            self.relation_property_tail = {x: [] for x in
                                             range(self.num_relation)}  # {relation_id:[tailid1, tailid2,...]}
            for t in self.triple_train:
                self.relation_property_head[t[1]].append(t[0])
                self.relation_property_tail[t[1]].append(t[2])  # t[1] -> tails

            self.relation_property = \
                {x: (len(set(self.relation_property_tail[x]))) /
                    (len(set(self.relation_property_head[x])) + len(set(self.relation_property_tail[x])))
                    for x in self.relation_property_head.keys()}
        # {relation_id: p, ...} 0< num <1, and for relation replace head entity with the property p
        else:
            print("unif set do'n need to calculate hpt and tph")

    def __init__(self):
        print("no attr")

    def __init__(self, data_dir, negative_sampling, learning_rate,
                 batch_size, max_iter, margin, dimension_e, dimension_r, norm):
        # this part for data prepare
        self.data_dir = data_dir
        self.negative_sampling = negative_sampling
        self.norm = norm

        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}

        self.triple_train = []  # [(head_id, relation_id, tail_id),...]
        self.triple_test = []
        self.triple_valid = []
        self.triple = []

        self.num_entity = 0
        self.num_relation = 0
        self.num_triple_train = 0
        self.num_triple_test = 0
        self.num_triple_valid = 0

        self.hr_t = defaultdict(set)
        self.tr_h = defaultdict(set)

        # load all the file: entity2id.txt, relation2id.txt, train.txt, test.txt, valid.txt
        self.load_data()
        print('finish preparing data. ')

        # this part for the model:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.margin = margin
        self.dimension_e = dimension_e
        self.dimension_r = dimension_r
        self.variables = []

    def get_loss(self, emb_p_h, emb_p_r, emb_p_t, emb_n_h, emb_n_r, emb_n_t):
        if self.norm == 'L1':
            score_positive = tf.reduce_sum(tf.abs(emb_p_h + emb_p_r - emb_p_t), axis=1)
            score_negative = tf.reduce_sum(tf.abs(emb_n_h + emb_n_r - emb_n_t), axis=1)
        elif self.norm == 'L2':
            score_positive = tf.reduce_sum((emb_p_h + emb_p_r - emb_p_t) ** 2, axis=1)
            score_negative = tf.reduce_sum((emb_n_h + emb_n_r - emb_n_t) ** 2, axis=1)
        else:
            raise NotImplementedError("Dose not support %s norm" % self.norm)

        loss = tf.reduce_sum(tf.maximum(0., score_positive + self.margin - score_negative))
        return loss


class PModel(object):
    def training_data_batch(self, batch_size):
        n_triple = len(self.triple_train)
        rand_idx = np.random.permutation(n_triple)  # 生成随机一个序列，用于打乱训练组
        start = 0
        while start < n_triple:
            start_t = timeit.default_timer()
            end = min(start + batch_size, n_triple)
            # size = end - start
            train_triple_positive = np.asarray([self.triple_train[x] for x in rand_idx[start:end]])
            train_triple_negative = []
            for t in train_triple_positive:
                replace_entity_id = np.random.randint(self.num_entity)
                random_num = np.random.random()

                if self.negative_sampling == 'unif':
                    replace_head_probability = 0.5
                elif self.negative_sampling == 'bern':
                    replace_head_probability = self.relation_property[t[1]]
                else:
                    raise NotImplementedError("Dose not support %s negative_sampling", self.negative_sampling)

                if random_num < replace_head_probability:
                    train_triple_negative.append((replace_entity_id, t[1], t[2]))
                else:
                    train_triple_negative.append((t[0], t[1], replace_entity_id))

            start = end
            prepare_t = timeit.default_timer() - start_t

            yield train_triple_positive, train_triple_negative, prepare_t

    def load_data(self):
        print('loading entity2id.txt ...')
        with open(os.path.join(self.data_dir, 'entity2id.txt')) as f:
            self.entity2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
            self.id2entity = {value: key for key, value in self.entity2id.items()}

        with open(os.path.join(self.data_dir, 'relation2id.txt')) as f:
            self.relation2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in
                                  f.readlines()}
            self.id2relation = {value: key for key, value in self.relation2id.items()}

        def load_triple(self, triplefile):
            triple_list = []  # [(head_id, relation_id, tail_id),...]
            with open(os.path.join(self.data_dir, triplefile)) as f:
                for line in f.readlines():
                    line_list = line.strip().split('\t')
                    assert len(line_list) == 3
                    headid = self.entity2id[line_list[0]]
                    relationid = self.relation2id[line_list[2]]
                    tailid = self.entity2id[line_list[1]]

                    triple_list.append((headid, relationid, tailid))  # h r t

                    self.hr_t[(headid, relationid)].add(tailid)
                    self.tr_h[(tailid, relationid)].add(headid)
            return triple_list

        self.triple_train = load_triple(self, 'train.txt')
        self.triple_test = load_triple(self, 'test.txt')
        self.triple_valid = load_triple(self, 'valid.txt')
        self.triple = np.concatenate([self.triple_train, self.triple_test, self.triple_valid], axis=0)  # triple 拼接

        self.num_relation = len(self.relation2id)
        self.num_entity = len(self.entity2id)
        self.num_triple_train = len(self.triple_train)
        self.num_triple_test = len(self.triple_test)
        self.num_triple_valid = len(self.triple_valid)

        print('entity number: ' + str(self.num_entity))
        print('relation number: ' + str(self.num_relation))
        print('training triple number: ' + str(self.num_triple_train))
        print('testing triple number: ' + str(self.num_triple_test))
        print('valid triple number: ' + str(self.num_triple_valid))

        if self.negative_sampling == 'bern':
            self.relation_property_head = {x: [] for x in
                                           range(self.num_relation)}  # {relation_id:[headid1, headid2,...]}
            self.relation_property_tail = {x: [] for x in
                                           range(self.num_relation)}  # {relation_id:[tailid1, tailid2,...]}
            for t in self.triple_train:
                self.relation_property_head[t[1]].append(t[0])
                self.relation_property_tail[t[1]].append(t[2])  # t[1] -> tails

            self.relation_property = {x:
                    ((len(set(self.relation_property_tail[x]))) / len(set(self.relation_property_tail[x]))) /
                    ((len(set(self.relation_property_head[x])) / len(set(self.relation_property_tail[x]))) +
                    (len(set(self.relation_property_tail[x])) / len(set(self.relation_property_head[x]))))
                 for x in self.relation_property_head.keys()}  # hpt
        else:
            print("unif set do'n need to calculate hpt and tph")

    def __init__(self):
        print("no attr")

    def __init__(self, data_dir, negative_sampling, learning_rate,
                 batch_size, max_iter, margin, dimension_e, dimension_r, norm):
        # this part for data prepare
        self.data_dir = data_dir
        self.negative_sampling = negative_sampling
        self.norm = norm

        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}

        self.triple_train = []  # [(head_id, relation_id, tail_id),...]
        self.triple_test = []
        self.triple_valid = []
        self.triple = []

        self.num_entity = 0
        self.num_relation = 0
        self.num_triple_train = 0
        self.num_triple_test = 0
        self.num_triple_valid = 0

        self.hr_t = defaultdict(set)
        self.tr_h = defaultdict(set)

        # load all the file: entity2id.txt, relation2id.txt, train.txt, test.txt, valid.txt
        self.load_data()
        print('finish preparing data. ')

        # this part for the model:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.margin = margin
        self.dimension_e = dimension_e
        self.dimension_r = dimension_r
        self.variables = []

    def get_loss(self, emb_p_h, emb_p_r, emb_p_t, emb_n_h, emb_n_r, emb_n_t):
        if self.norm == 'L1':
            score_positive = tf.reduce_sum(tf.abs(emb_p_h + emb_p_r - emb_p_t), axis=1)
            score_negative = tf.reduce_sum(tf.abs(emb_n_h + emb_n_r - emb_n_t), axis=1)
        elif self.norm == 'L2':
            score_positive = tf.reduce_sum((emb_p_h + emb_p_r - emb_p_t) ** 2, axis=1)
            score_negative = tf.reduce_sum((emb_n_h + emb_n_r - emb_n_t) ** 2, axis=1)
        else:
            raise NotImplementedError("Dose not support %s norm" % self.norm)

        loss = tf.reduce_sum(tf.maximum(0., score_positive + self.margin - score_negative))
        return loss


class PModelEX(object):
    def training_data_batch(self, batch_size):
        n_triple = len(self.triple_train)
        rand_idx = np.random.permutation(n_triple)  # 生成随机一个序列，用于打乱训练组
        start = 0
        while start < n_triple:
            start_t = timeit.default_timer()
            end = min(start + batch_size, n_triple)
            # size = end - start
            train_triple_positive = np.asarray([self.triple_train[x] for x in rand_idx[start:end]])
            train_triple_negative = []
            for t in train_triple_positive:
                replace_entity_id = np.random.randint(self.num_entity)
                random_num = np.random.random()

                if self.negative_sampling == 'unif':
                    replace_head_probability = 0.5
                elif self.negative_sampling == 'bern':
                    replace_head_probability = self.relation_property[t[1]]
                else:
                    raise NotImplementedError("Dose not support %s negative_sampling", self.negative_sampling)

                if random_num < replace_head_probability:
                    train_triple_negative.append((replace_entity_id, t[1], t[2]))
                else:
                    train_triple_negative.append((t[0], t[1], replace_entity_id))

            start = end
            prepare_t = timeit.default_timer() - start_t

            yield train_triple_positive, train_triple_negative, prepare_t

    def load_data(self):
        print('loading entity2id.txt ...')
        with open(os.path.join(self.data_dir, 'entity2id.txt')) as f:
            self.entity2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
            self.id2entity = {value: key for key, value in self.entity2id.items()}

        with open(os.path.join(self.data_dir, 'relation2id.txt')) as f:
            self.relation2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in
                                  f.readlines()}
            self.id2relation = {value: key for key, value in self.relation2id.items()}

        def load_triple(self, triplefile):
            triple_list = []  # [(head_id, relation_id, tail_id),...]
            with open(os.path.join(self.data_dir, triplefile)) as f:
                for line in f.readlines():
                    line_list = line.strip().split('\t')
                    assert len(line_list) == 3
                    headid = self.entity2id[line_list[0]]
                    relationid = self.relation2id[line_list[1]]
                    tailid = self.entity2id[line_list[2]]

                    triple_list.append((headid, relationid, tailid))  # h r t

                    self.hr_t[(headid, relationid)].add(tailid)
                    self.tr_h[(tailid, relationid)].add(headid)
            return triple_list

        self.triple_train = load_triple(self, 'train.txt')
        self.triple_test = load_triple(self, 'test.txt')
        self.triple_valid = load_triple(self, 'valid.txt')
        self.triple = np.concatenate([self.triple_train, self.triple_test, self.triple_valid], axis=0)  # triple 拼接

        self.num_relation = len(self.relation2id)
        self.num_entity = len(self.entity2id)
        self.num_triple_train = len(self.triple_train)
        self.num_triple_test = len(self.triple_test)
        self.num_triple_valid = len(self.triple_valid)

        print('entity number: ' + str(self.num_entity))
        print('relation number: ' + str(self.num_relation))
        print('training triple number: ' + str(self.num_triple_train))
        print('testing triple number: ' + str(self.num_triple_test))
        print('valid triple number: ' + str(self.num_triple_valid))

        if self.negative_sampling == 'bern':
            self.relation_property_head = {x: [] for x in
                                           range(self.num_relation)}  # {relation_id:[headid1, headid2,...]}
            self.relation_property_tail = {x: [] for x in
                                           range(self.num_relation)}  # {relation_id:[tailid1, tailid2,...]}
            for t in self.triple_train:
                self.relation_property_head[t[1]].append(t[0])
                self.relation_property_tail[t[1]].append(t[2])  # t[1] -> tails

            self.relation_property = {x:
                    ((len(set(self.relation_property_tail[x]))) / len(set(self.relation_property_tail[x]))) /
                    ((len(set(self.relation_property_head[x])) / len(set(self.relation_property_tail[x]))) +
                    (len(set(self.relation_property_tail[x])) / len(set(self.relation_property_head[x]))))
                 for x in self.relation_property_head.keys()}  # hpt
        else:
            print("unif set do'n need to calculate hpt and tph")

    def __init__(self):
        print("no attr")

    def __init__(self, data_dir, negative_sampling, learning_rate,
                 batch_size, max_iter, margin, dimension_e, dimension_r, norm):
        # this part for data prepare
        self.data_dir = data_dir
        self.negative_sampling = negative_sampling
        self.norm = norm

        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}

        self.triple_train = []  # [(head_id, relation_id, tail_id),...]
        self.triple_test = []
        self.triple_valid = []
        self.triple = []

        self.num_entity = 0
        self.num_relation = 0
        self.num_triple_train = 0
        self.num_triple_test = 0
        self.num_triple_valid = 0

        self.hr_t = defaultdict(set)
        self.tr_h = defaultdict(set)

        # load all the file: entity2id.txt, relation2id.txt, train.txt, test.txt, valid.txt
        self.load_data()
        print('finish preparing data. ')

        # this part for the model:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.margin = margin
        self.dimension_e = dimension_e
        self.dimension_r = dimension_r
        self.variables = []

    def get_loss(self, emb_p_h, emb_p_r, emb_p_t, emb_n_h, emb_n_r, emb_n_t):
        if self.norm == 'L1':
            score_positive = tf.reduce_sum(tf.abs(emb_p_h + emb_p_r - emb_p_t), axis=1)
            score_negative = tf.reduce_sum(tf.abs(emb_n_h + emb_n_r - emb_n_t), axis=1)
        elif self.norm == 'L2':
            score_positive = tf.reduce_sum((emb_p_h + emb_p_r - emb_p_t) ** 2, axis=1)
            score_negative = tf.reduce_sum((emb_n_h + emb_n_r - emb_n_t) ** 2, axis=1)
        else:
            raise NotImplementedError("Dose not support %s norm" % self.norm)

        loss = tf.reduce_sum(tf.maximum(0., score_positive + self.margin - score_negative))
        return loss


class TransE(PModelEX):  # rate:0.01/0.001 dimension:100,100 batch:4800 norm:L2 margin:1.0
    def __init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                 dimension_r, norm):

        PModel.__init__(self, data_dir, negative_sampling, learning_rate,batch_size, max_iter, margin, dimension_e, dimension_r, norm)

        bound = 6 / math.sqrt(self.dimension_e)

        with tf.device('/gpu'):
            self.embedding_entity = tf.get_variable('embedding_entity', [self.num_entity, self.dimension_r],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,maxval=bound))
            self.embedding_relation = tf.get_variable('embedding_relation', [self.num_relation, self.dimension_r],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,maxval=bound))
            self.variables.append(self.embedding_entity)
            self.variables.append(self.embedding_relation)  # 随机初始化向量
            print('finishing initializing')

    def train(self, inputs):
        embedding_relation = self.embedding_relation
        embedding_entity = self.embedding_entity

        triple_positive, triple_negative = inputs  # triple_positive:(head_id,relation_id,tail_id)

        norm_entity = tf.nn.l2_normalize(embedding_entity, dim=1)  # 按行求L2标准化 shape=(14951, 100), dtype=float32
        norm_relation = tf.nn.l2_normalize(embedding_relation, dim=1)

        emb_p_h = tf.nn.embedding_lookup(norm_entity, triple_positive[:, 0])  # 取第0列的所有元素,即获取正样本中所有的头向量
        emb_p_t = tf.nn.embedding_lookup(norm_entity, triple_positive[:, 2])
        emb_p_r = tf.nn.embedding_lookup(norm_relation, triple_positive[:, 1])

        emb_n_h = tf.nn.embedding_lookup(norm_entity, triple_negative[:, 0])
        emb_n_t = tf.nn.embedding_lookup(norm_entity, triple_negative[:, 2])
        emb_n_r = tf.nn.embedding_lookup(norm_relation, triple_negative[:, 1])

        return PModel.get_loss(self, emb_p_h, emb_p_r, emb_p_t, emb_n_h, emb_n_r, emb_n_t)

    def test(self, inputs):
        embedding_relation = self.embedding_relation
        embedding_entity = self.embedding_entity

        triple_test = inputs  # (headid, tailid, tailid)

        norm_emb_e = tf.nn.l2_normalize(embedding_entity, dim=1)
        norm_emb_r = tf.nn.l2_normalize(embedding_relation, dim=1)
        norm_vec_h = tf.nn.embedding_lookup(norm_emb_e, triple_test[0])
        norm_vec_r = tf.nn.embedding_lookup(norm_emb_r, triple_test[1])
        norm_vec_t = tf.nn.embedding_lookup(norm_emb_e, triple_test[2])

        if self.norm == 'L1':
            _, norm_id_replace_head = tf.nn.top_k(
                tf.reduce_sum(tf.abs(norm_emb_e + norm_vec_r - norm_vec_t), axis=1), k=self.num_entity)
            _, norm_id_replace_tail = tf.nn.top_k(
                tf.reduce_sum(tf.abs(norm_vec_h + norm_vec_r - norm_emb_e), axis=1), k=self.num_entity)
        elif self.norm == 'L2':
            _, norm_id_replace_head = tf.nn.top_k(
                tf.reduce_sum((norm_emb_e + norm_vec_r - norm_vec_t)**2, axis=1),
                k=self.num_entity)
            _, norm_id_replace_tail = tf.nn.top_k(
                tf.reduce_sum((norm_vec_h + norm_vec_r - norm_emb_e)**2, axis=1),
                k=self.num_entity)
        else:
            raise NotImplementedError("Dose not support %s norm" % self.norm)

        return norm_id_replace_head, norm_id_replace_tail


class TransR(PModel):  # rate:0.001 dimension:100,100 batch:1000 norm:L2 margin:1.0
    def __init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                 dimension_r, norm):

        PModel.__init__(self, data_dir, negative_sampling, learning_rate,batch_size, max_iter, margin, dimension_e, dimension_r, norm)

        bound = 6 / math.sqrt(self.dimension_e)

        with tf.device('/gpu'):
            self.embedding_entity = tf.get_variable('embedding_entity', [self.num_entity, self.dimension_r],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,maxval=bound))
            self.embedding_relation = tf.get_variable('embedding_relation', [self.num_relation, self.dimension_r],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,maxval=bound))
            self.variables.append(self.embedding_entity)
            self.variables.append(self.embedding_relation)
            rel_matrix = np.zeros([self.num_relation, self.dimension_e * self.dimension_r], dtype=np.float32)
            for i in range(self.num_relation):
                for j in range(self.dimension_e):
                    for k in range(self.dimension_r):
                        if j == k:
                            rel_matrix[i][j * self.dimension_r + k] = 1.0
            self.relation_matrix = tf.Variable(rel_matrix, name="rel_matrix")  # 随机初始化向量
            print('finishing initializing')

    def train(self, inputs):
        embedding_relation = self.embedding_relation
        embedding_entity = self.embedding_entity
        relation_matrix = self.relation_matrix
        size_e = self.dimension_e
        size_r = self.dimension_r

        triple_positive, triple_negative = inputs  # triple_positive:(head_id,relation_id,tail_id)

        norm_entity = tf.nn.l2_normalize(embedding_entity, dim=1)  # 按行求L2标准化 shape=(14951, 100), dtype=float32
        norm_relation = tf.nn.l2_normalize(embedding_relation, dim=1)

        emb_p_h = tf.reshape(tf.nn.embedding_lookup(norm_entity, triple_positive[:, 0]),[-1,1,size_e])  # 取第0列的所有元素,即获取正样本中所有的头向量
        emb_p_t = tf.reshape(tf.nn.embedding_lookup(norm_entity, triple_positive[:, 2]),[-1,1,size_e])
        emb_p_r = tf.reshape(tf.nn.embedding_lookup(norm_relation, triple_positive[:, 1]),[-1,size_r])
        emb_p_m = tf.reshape(tf.nn.embedding_lookup(relation_matrix, triple_positive[:, 1]),[-1,size_e,size_r])

        emb_n_h = tf.reshape(tf.nn.embedding_lookup(norm_entity, triple_negative[:, 0]),[-1,1,size_e])
        emb_n_t = tf.reshape(tf.nn.embedding_lookup(norm_entity, triple_negative[:, 2]),[-1,1,size_e])
        emb_n_r = tf.reshape(tf.nn.embedding_lookup(norm_relation, triple_negative[:, 1]),[-1,size_r])
        emb_n_m = tf.reshape(tf.nn.embedding_lookup(relation_matrix, triple_negative[:, 1]),[-1,size_e,size_r])

        emb_p_h = tf.reshape(tf.matmul(emb_p_h,emb_p_m),[-1,size_r])
        emb_n_h = tf.reshape(tf.matmul(emb_n_h,emb_n_m),[-1,size_r])
        emb_p_t = tf.reshape(tf.matmul(emb_p_t,emb_p_m),[-1,size_r])
        emb_n_t = tf.reshape(tf.matmul(emb_n_t,emb_n_m),[-1,size_r])

        emb_p_h = tf.nn.l2_normalize(emb_p_h,1)
        emb_n_h = tf.nn.l2_normalize(emb_n_h,1)
        emb_p_t = tf.nn.l2_normalize(emb_p_t,1)
        emb_n_t = tf.nn.l2_normalize(emb_n_t,1)

        return PModel.get_loss(self, emb_p_h, emb_p_r, emb_p_t, emb_n_h, emb_n_r, emb_n_t)

    def test(self, inputs):
        embedding_relation = self.embedding_relation
        embedding_entity = self.embedding_entity
        relation_matrix = self.relation_matrix
        size_e = self.dimension_e
        size_r = self.dimension_r

        triple_test = inputs

        norm_emb_e = tf.nn.l2_normalize(embedding_entity, dim=1)
        norm_emb_r = tf.nn.l2_normalize(embedding_relation, dim=1)

        vec_h = tf.reshape(tf.nn.embedding_lookup(norm_emb_e, triple_test[0]), [1, size_e])
        vec_r = tf.reshape(tf.nn.embedding_lookup(norm_emb_r, triple_test[1]), [1, size_r])
        vec_t = tf.reshape(tf.nn.embedding_lookup(norm_emb_e, triple_test[2]), [1, size_e])
        matrix_vec = tf.reshape(tf.nn.embedding_lookup(relation_matrix, triple_test[1]),[size_e, size_r])

        vec_h = tf.nn.l2_normalize(tf.reshape(tf.matmul(vec_h, matrix_vec), [1,size_r]), 1)
        vec_t = tf.nn.l2_normalize(tf.reshape(tf.matmul(vec_t, matrix_vec), [1,size_r]), 1)
        # vec_r = tf.nn.l2_normalize(tf.reshape(vec_r, [1,sizeR]), 1)

        replace_entity = tf.nn.l2_normalize(tf.reshape(tf.matmul(norm_emb_e, matrix_vec), [-1,size_r]), 1)

        if self.norm == 'L1':
            _, norm_id_replace_head = tf.nn.top_k(
                tf.reduce_sum(tf.abs(replace_entity + vec_r - vec_t), axis=1), k=self.num_entity)
            _, norm_id_replace_tail = tf.nn.top_k(
                tf.reduce_sum(tf.abs(vec_h + vec_r - replace_entity), axis=1), k=self.num_entity)
        elif self.norm == 'L2':
            _, norm_id_replace_head = tf.nn.top_k(
                tf.reduce_sum((replace_entity + vec_r - vec_t)**2, axis=1),
                k=self.num_entity)
            _, norm_id_replace_tail = tf.nn.top_k(
                tf.reduce_sum((vec_h + vec_r - replace_entity)**2, axis=1),
                k=self.num_entity)
        else:
            raise NotImplementedError("Dose not support %s norm" % self.norm)

        return norm_id_replace_head, norm_id_replace_tail


class TransH(PModel): # rate:0.001 dimension:100,100 batch:4800 norm:L2 margin:1.0
    def __init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                 dimension_r, norm):

        PModel.__init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                        dimension_r, norm)

        bound = 6 / math.sqrt(self.dimension_e)

        with tf.device('/gpu'):
            self.embedding_entity = tf.get_variable('embedding_entity', [self.num_entity, self.dimension_r],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            self.embedding_relation = tf.get_variable('embedding_relation', [self.num_relation, self.dimension_r],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            self.embedding_hyperplanes = tf.get_variable('embedding_hyperplanes', [self.num_relation, self.dimension_r],
                                                         initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                   maxval=bound))
            self.variables.append(self.embedding_entity)
            self.variables.append(self.embedding_relation)
            self.variables.append(self.embedding_hyperplanes)  # 随机初始化向量
            print('finishing initializing')

    def train(self, inputs):
        with tf.device('/gpu'):
            embedding_relation = self.embedding_relation
            embedding_entity = self.embedding_entity
            embedding_hyperplanes = self.embedding_hyperplanes

            triple_positive, triple_negative = inputs  # triple_positive:(head_id,relation_id,tail_id)

            norm_entity = tf.nn.l2_normalize(embedding_entity, dim=1)  # 按行求L2标准化 shape=(14951, 100), dtype=float32
            norm_relation = tf.nn.l2_normalize(embedding_relation, dim=1)
            norm_hyperplanes = tf.nn.l2_normalize(embedding_hyperplanes, dim=1)  # 将第2维-将列求和,计算得到每个向量的L2范数 # nothing useful

            emb_p_h = tf.nn.embedding_lookup(norm_entity, triple_positive[:, 0])  # 取第0列的所有元素,即获取正样本中所有的头向量
            emb_p_t = tf.nn.embedding_lookup(norm_entity, triple_positive[:, 2])
            emb_p_r = tf.nn.embedding_lookup(norm_relation, triple_positive[:, 1])
            emb_p_hp = tf.nn.embedding_lookup(norm_hyperplanes, triple_positive[:, 1])

            emb_n_h = tf.nn.embedding_lookup(norm_entity, triple_negative[:, 0])
            emb_n_t = tf.nn.embedding_lookup(norm_entity, triple_negative[:, 2])
            emb_n_r = tf.nn.embedding_lookup(norm_relation, triple_negative[:, 1])
            emb_n_hp = tf.nn.embedding_lookup(norm_hyperplanes, triple_negative[:, 1])

            emb_p_h = emb_p_h - emb_p_hp * tf.reduce_sum(emb_p_h * emb_p_hp, 1, keep_dims=True)
            emb_n_h = emb_n_h - emb_n_hp * tf.reduce_sum(emb_n_h * emb_n_hp, 1, keep_dims=True)
            emb_p_t = emb_p_t - emb_p_hp * tf.reduce_sum(emb_p_t * emb_p_hp, 1, keep_dims=True)
            emb_n_t = emb_n_t - emb_n_hp * tf.reduce_sum(emb_n_t * emb_n_hp, 1, keep_dims=True)

            return PModel.get_loss(self, emb_p_h, emb_p_r, emb_p_t, emb_n_h, emb_n_r, emb_n_t)

    def test(self, inputs):
        with tf.device('/gpu'):
            embedding_relation = self.embedding_relation
            embedding_entity = self.embedding_entity
            embedding_hyperplanes = self.embedding_hyperplanes

            triple_test = inputs  # (headid, tailid, tailid)

            norm_emb_e = tf.nn.l2_normalize(embedding_entity, dim=1)
            norm_embedding_hyperplanes = tf.nn.l2_normalize(embedding_hyperplanes, dim=1)
            norm_emb_r = tf.nn.l2_normalize(embedding_relation, dim=1)
            norm_vec_h = tf.nn.embedding_lookup(norm_emb_e, triple_test[0])
            norm_vec_r = tf.nn.embedding_lookup(norm_emb_r, triple_test[1])
            norm_vec_t = tf.nn.embedding_lookup(norm_emb_e, triple_test[2])
            norm_hyperplanes_vec = tf.nn.embedding_lookup(norm_embedding_hyperplanes, triple_test[1])

            norm_vec_h = norm_vec_h - norm_hyperplanes_vec * tf.reduce_sum(norm_vec_h * norm_hyperplanes_vec)
            norm_vec_t = norm_vec_t - norm_hyperplanes_vec * tf.reduce_sum(norm_vec_t * norm_hyperplanes_vec)

            replace_embedding = embedding_entity - norm_hyperplanes_vec * tf.reduce_sum(
                embedding_entity * norm_hyperplanes_vec, keep_dims=True)
            norm_replace_embedding = tf.nn.l2_normalize(replace_embedding, 1)

            if self.norm == 'L1':
                _, norm_id_replace_head = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(norm_replace_embedding + norm_vec_r - norm_vec_t), axis=1),
                    k=self.num_entity)
                _, norm_id_replace_tail = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(norm_vec_h + norm_vec_r - norm_replace_embedding), axis=1),
                    k=self.num_entity)
            elif self.norm == 'L2':
                _, norm_id_replace_head = tf.nn.top_k(
                    tf.reduce_sum((norm_replace_embedding + norm_vec_r - norm_vec_t) ** 2, axis=1),
                    k=self.num_entity)
                _, norm_id_replace_tail = tf.nn.top_k(
                    tf.reduce_sum((norm_vec_h + norm_vec_r - norm_replace_embedding) ** 2, axis=1),
                    k=self.num_entity)
            else:
                raise NotImplementedError("Dose not support %s norm" % self.norm)

            return norm_id_replace_head, norm_id_replace_tail


class TransD(PModel):  # rate:0.001 dimension:50,50 batch:200 norm:L2 margin:1.0
    def __init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                 dimension_r, norm):

        PModel.__init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                        dimension_r, norm)

        bound = 6 / math.sqrt(self.dimension_e)

        with tf.device('/gpu'):
            self.embedding_entity = tf.get_variable('embedding_entity', [self.num_entity, self.dimension_r],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            self.embedding_relation = tf.get_variable('embedding_relation', [self.num_relation, self.dimension_r],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            self.projection_entity = tf.get_variable('projection_entity', [self.num_entity, self.dimension_e],
                                                     initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                 maxval=bound))
            self.projection_relation = tf.get_variable('projection_relation',
                                                       [self.num_relation, self.dimension_r],
                                                       initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                 maxval=bound))
            self.variables.append(self.embedding_entity)
            self.variables.append(self.embedding_relation)
            self.variables.append(self.projection_entity)
            self.variables.append(self.projection_relation)  # 随机初始化向量
            print('finishing initializing')

    def train(self, inputs):
        with tf.device('/gpu'):
            embedding_relation = self.embedding_relation
            embedding_entity = self.embedding_entity
            projection_relation = self.projection_relation
            projection_entity = self.projection_entity

            triple_positive, triple_negative = inputs  # triple_positive:(head_id,relation_id,tail_id)

            norm_emb_entity = tf.nn.l2_normalize(embedding_entity, dim=1)  # 按行求L2标准化 shape=(14951, 100), dtype=float32
            norm_emb_relation = tf.nn.l2_normalize(embedding_relation, dim=1)
            norm_pro_entity = tf.nn.l2_normalize(projection_entity, dim=1)
            norm_pro_relation = tf.nn.l2_normalize(projection_relation, dim=1)

            emb_p_h = tf.nn.embedding_lookup(norm_emb_entity, triple_positive[:, 0])  # 取第0列的所有元素,即获取正样本中所有的头向量
            emb_p_t = tf.nn.embedding_lookup(norm_emb_entity, triple_positive[:, 2])
            emb_p_r = tf.nn.embedding_lookup(norm_emb_relation, triple_positive[:, 1])
            pro_p_h = tf.nn.embedding_lookup(norm_pro_entity, triple_positive[:, 0])
            pro_p_t = tf.nn.embedding_lookup(norm_pro_entity, triple_positive[:, 2])
            pro_p_r = tf.nn.embedding_lookup(norm_pro_relation, triple_positive[:, 1])

            emb_n_h = tf.nn.embedding_lookup(norm_emb_entity, triple_negative[:, 0])
            emb_n_t = tf.nn.embedding_lookup(norm_emb_entity, triple_negative[:, 2])
            emb_n_r = tf.nn.embedding_lookup(norm_emb_relation, triple_negative[:, 1])
            pro_n_h = tf.nn.embedding_lookup(norm_pro_entity, triple_negative[:, 0])
            pro_n_t = tf.nn.embedding_lookup(norm_pro_entity, triple_negative[:, 2])
            pro_n_r = tf.nn.embedding_lookup(norm_pro_relation, triple_negative[:, 1])

            f_emb_p_h = tf.nn.l2_normalize(emb_p_h + tf.reduce_sum(emb_p_h * pro_p_h, 1, keep_dims=True) * pro_p_r, 1)
            f_emb_n_h = tf.nn.l2_normalize(emb_n_h + tf.reduce_sum(emb_n_h * pro_n_h, 1, keep_dims=True) * pro_n_r, 1)
            f_emb_p_t = tf.nn.l2_normalize(emb_p_t + tf.reduce_sum(emb_p_t * pro_p_t, 1, keep_dims=True) * pro_p_r, 1)
            f_emb_n_t = tf.nn.l2_normalize(emb_n_t + tf.reduce_sum(emb_n_t * pro_n_t, 1, keep_dims=True) * pro_n_r, 1)
            f_emb_p_r = tf.nn.l2_normalize(emb_p_r, 1)
            f_emb_n_r = tf.nn.l2_normalize(emb_n_r, 1)

            return PModel.get_loss(self, f_emb_p_h, f_emb_p_r, f_emb_p_t, f_emb_n_h, f_emb_n_r, f_emb_n_t)

    def test(self, inputs):
        with tf.device('/gpu'):
            embedding_relation = self.embedding_relation
            embedding_entity = self.embedding_entity
            projection_relation = self.projection_relation
            projection_entity = self.projection_entity

            triple_test = inputs

            norm_emb_entity = tf.nn.l2_normalize(embedding_entity, dim=1)
            norm_emb_relation = tf.nn.l2_normalize(embedding_relation, dim=1)
            norm_pro_entity = tf.nn.l2_normalize(projection_entity, dim=1)
            norm_pro_relation = tf.nn.l2_normalize(projection_relation, dim=1)

            emb_h = tf.nn.embedding_lookup(norm_emb_entity, triple_test[0])
            emb_r = tf.nn.embedding_lookup(norm_emb_relation, triple_test[1])
            emb_t = tf.nn.embedding_lookup(norm_emb_entity, triple_test[2])
            pro_h = tf.nn.embedding_lookup(norm_pro_entity, triple_test[0])
            pro_r = tf.nn.embedding_lookup(norm_pro_relation, triple_test[1])
            pro_t = tf.nn.embedding_lookup(norm_pro_entity, triple_test[2])

            f_emb_h = tf.nn.l2_normalize(emb_h + tf.reduce_sum(emb_h * pro_h) * pro_r)
            f_emb_t = tf.nn.l2_normalize(emb_t + tf.reduce_sum(emb_t * pro_t) * pro_r)
            f_emb_r = tf.nn.l2_normalize(emb_r)

            replace_entity = norm_emb_entity + tf.reduce_sum(norm_emb_entity * norm_pro_entity, 1, keep_dims=True) * pro_r
            replace_entity = tf.nn.l2_normalize(replace_entity, 1)

            if self.norm == 'L1':
                _, norm_id_replace_head = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(replace_entity + f_emb_r - f_emb_t), axis=1), k=self.__num_entity)
                _, norm_id_replace_tail = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(f_emb_h + f_emb_r - replace_entity), axis=1), k=self.__num_entity)
            elif self.norm == 'L2':
                _, norm_id_replace_head = tf.nn.top_k(
                    tf.reduce_sum((replace_entity + f_emb_r - f_emb_t) ** 2, axis=1), k=self.num_entity)
                _, norm_id_replace_tail = tf.nn.top_k(
                    tf.reduce_sum((f_emb_h + f_emb_r - replace_entity) ** 2, axis=1), k=self.num_entity)
            else:
                raise NotImplementedError("Dose not support %s norm" % self.norm)

            return norm_id_replace_head, norm_id_replace_tail


class TransMS(PModelEX):  # rate:0.001 dimension:200,200 batch:4800 norm:L1 margin:2.0
    def __init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                 dimension_r, norm):

        PModel.__init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                        dimension_r, norm)

        bound = 6 / math.sqrt(self.dimension_e)

        with tf.device('/gpu'):
            self.embedding_entity = tf.get_variable('embedding_entity', [self.num_entity, self.dimension_r],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            self.embedding_relation = tf.get_variable('embedding_relation', [self.num_relation, self.dimension_r],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            self.parameter_relation = tf.get_variable('parameter_relation', [self.num_relation, 1],
                                                      initializer=tf.random_uniform_initializer(minval=0,
                                                                                                maxval=0)
                                                      )
            self.variables.append(self.embedding_entity)
            self.variables.append(self.embedding_relation)
            self.variables.append(self.parameter_relation)  # 随机初始化向量
            print('finishing initializing')

    def train(self, inputs):
        with tf.device('/gpu'):
            embedding_relation = self.embedding_relation
            embedding_entity = self.embedding_entity
            parameter_relation = self.parameter_relation

            triple_positive, triple_negative = inputs  # triple_positive:(head_id,relation_id,tail_id)

            norm_entity = tf.nn.l2_normalize(embedding_entity, dim=1)  # 按行求L2标准化 shape=(14951, 100), dtype=float32
            norm_relation = tf.nn.l2_normalize(embedding_relation, dim=1)
            # norm_parameter = tf.nn.l2_normalize(parameter_relation, dim=1)
            # norm_parameter = parameter_relation

            emb_p_h = tf.nn.embedding_lookup(norm_entity, triple_positive[:, 0])  # 取第0列的所有元素,即获取正样本中所有的头向量
            emb_p_t = tf.nn.embedding_lookup(norm_entity, triple_positive[:, 2])
            emb_p_r = tf.nn.embedding_lookup(norm_relation, triple_positive[:, 1])
            emb_p_p = tf.nn.embedding_lookup(parameter_relation, triple_positive[:, 1])

            emb_n_h = tf.nn.embedding_lookup(norm_entity, triple_negative[:, 0])
            emb_n_t = tf.nn.embedding_lookup(norm_entity, triple_negative[:, 2])
            emb_n_r = tf.nn.embedding_lookup(norm_relation, triple_negative[:, 1])
            emb_n_p = tf.nn.embedding_lookup(parameter_relation, triple_negative[:, 1])

            f_emb_p_h = -tf.tanh(emb_p_t * emb_p_r) * emb_p_h
            f_emb_n_h = -tf.tanh(emb_n_t * emb_n_r) * emb_n_h
            f_emb_p_t = tf.tanh(emb_p_h * emb_p_r) * emb_p_t
            f_emb_n_t = tf.tanh(emb_n_h * emb_n_r) * emb_n_t
            f_emb_p_r = emb_p_r + emb_p_p * emb_p_h * emb_p_t
            f_emb_n_r = emb_n_r + emb_n_p * emb_n_h * emb_n_t

            return PModel.get_loss(self, f_emb_p_h, f_emb_p_r, f_emb_p_t, f_emb_n_h, f_emb_n_r, f_emb_n_t)

    def test(self, inputs):
        with tf.device('/gpu'):
            embedding_relation = self.embedding_relation
            embedding_entity = self.embedding_entity
            parameter_relation = self.parameter_relation

            triple_test = inputs  # (headid, tailid, tailid)

            norm_emb_e = tf.nn.l2_normalize(embedding_entity, dim=1)
            # norm_para_r = tf.nn.l2_normalize(parameter_relation, dim=1)
            # norm_para_relation = parameter_relation
            norm_emb_r = tf.nn.l2_normalize(embedding_relation, dim=1)
            norm_vec_h = tf.nn.embedding_lookup(norm_emb_e, triple_test[0])
            norm_vec_t = tf.nn.embedding_lookup(norm_emb_e, triple_test[2])
            norm_vec_r = tf.nn.embedding_lookup(norm_emb_r, triple_test[1])
            norm_parameter = tf.nn.embedding_lookup(parameter_relation, triple_test[1])

            f_norm_vec_h = -tf.tanh(norm_vec_t * norm_vec_r) * norm_vec_h
            f_norm_vec_t = tf.tanh(norm_vec_h * norm_vec_r) * norm_vec_t

            replace_emb_h = -tf.tanh(norm_vec_t * norm_vec_r) * norm_emb_e
            replace_emb_t = tf.tanh(norm_vec_t * norm_vec_r) * norm_emb_e
            replace_emb_r_h = norm_vec_r + norm_parameter * norm_emb_e * norm_vec_t
            replace_emb_r_t = norm_vec_r + norm_parameter * norm_vec_h * norm_emb_e

            if self.norm == 'L1':
                _, norm_id_replace_head = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(replace_emb_h + replace_emb_r_h - f_norm_vec_t), axis=1),
                    k=self.num_entity)
                _, norm_id_replace_tail = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(f_norm_vec_h + replace_emb_r_t - replace_emb_t), axis=1),
                    k=self.num_entity)
            elif self.norm == 'L2':
                _, norm_id_replace_head = tf.nn.top_k(
                    tf.reduce_sum((replace_emb_h + replace_emb_r_h - f_norm_vec_t) ** 2, axis=1),
                    k=self.num_entity)
                _, norm_id_replace_tail = tf.nn.top_k(
                    tf.reduce_sum((f_norm_vec_h + replace_emb_r_t - replace_emb_t) ** 2, axis=1),
                    k=self.num_entity)
            else:
                raise NotImplementedError("Dose not support %s norm" % self.norm)

            return norm_id_replace_head, norm_id_replace_tail


class TransEP(PModel):  # rate:0.01/0.001 dimension:100,100 batch:4800 norm:L2 margin:1.0
    def __init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                 dimension_r, norm):

        PModel.__init__(self, data_dir, negative_sampling, learning_rate,batch_size, max_iter, margin, dimension_e, dimension_r, norm)

        bound = 6 / math.sqrt(self.dimension_e)

        with tf.device('/gpu'):
            self.embedding_entity_head = tf.get_variable('embedding_entity_head', [self.num_entity, self.dimension_e],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            self.embedding_entity_tail = tf.get_variable('embedding_entity_tail', [self.num_entity, self.dimension_e],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            self.embedding_relation = tf.get_variable('embedding_relation', [self.num_relation, self.dimension_r],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            self.variables.append(self.embedding_entity_head)
            self.variables.append(self.embedding_entity_tail)
            self.variables.append(self.embedding_relation)  # 随机初始化向量
            print('finishing initializing')

    def train(self, inputs):
        embedding_relation = self.embedding_relation
        embedding_entity_head = self.embedding_entity_head
        embedding_entity_tail = self.embedding_entity_tail

        triple_positive, triple_negative = inputs  # triple_positive:(head_id,relation_id,tail_id)

        norm_entity_head = tf.nn.l2_normalize(embedding_entity_head, dim=1)  # 按行求L2标准化 shape=(14951, 100), dtype=float32
        norm_entity_tail = tf.nn.l2_normalize(embedding_entity_tail, dim=1)
        norm_relation = tf.nn.l2_normalize(embedding_relation, dim=1)

        emb_p_h = tf.nn.embedding_lookup(norm_entity_head, triple_positive[:, 0])  # 取第0列的所有元素,即获取正样本中所有的头向量
        emb_p_t = tf.nn.embedding_lookup(norm_entity_tail, triple_positive[:, 2])
        emb_p_r = tf.nn.embedding_lookup(norm_relation, triple_positive[:, 1])

        emb_n_h = tf.nn.embedding_lookup(norm_entity_head, triple_negative[:, 0])
        emb_n_t = tf.nn.embedding_lookup(norm_entity_tail, triple_negative[:, 2])
        emb_n_r = tf.nn.embedding_lookup(norm_relation, triple_negative[:, 1])

        return PModel.get_loss(self, emb_p_h, emb_p_r, emb_p_t, emb_n_h, emb_n_r, emb_n_t)

    def test(self, inputs):
        embedding_relation = self.embedding_relation
        embedding_entity_head = self.embedding_entity_head
        embedding_entity_tail = self.embedding_entity_tail

        triple_test = inputs  # (headid, tailid, tailid)

        norm_emb_e_head = tf.nn.l2_normalize(embedding_entity_head, dim=1)
        norm_emb_e_tail = tf.nn.l2_normalize(embedding_entity_tail, dim=1)
        norm_emb_r = tf.nn.l2_normalize(embedding_relation, dim=1)
        norm_vec_h = tf.nn.embedding_lookup(norm_emb_e_head, triple_test[0])
        norm_vec_r = tf.nn.embedding_lookup(norm_emb_r, triple_test[1])
        norm_vec_t = tf.nn.embedding_lookup(norm_emb_e_tail, triple_test[2])

        if self.norm == 'L1':
            _, norm_id_replace_head = tf.nn.top_k(
                tf.reduce_sum(tf.abs(norm_emb_e_head + norm_vec_r - norm_vec_t), axis=1),
                k=self.num_entity)
            _, norm_id_replace_tail = tf.nn.top_k(
                tf.reduce_sum(tf.abs(norm_vec_h + norm_vec_r - norm_emb_e_tail), axis=1),
                k=self.num_entity)
        elif self.norm == 'L2':
            _, norm_id_replace_head = tf.nn.top_k(
                tf.reduce_sum((norm_emb_e_head + norm_vec_r - norm_vec_t)**2, axis=1),
                k=self.num_entity)
            _, norm_id_replace_tail = tf.nn.top_k(
                tf.reduce_sum((norm_vec_h + norm_vec_r - norm_emb_e_tail)**2, axis=1),
                k=self.num_entity)
        else:
            raise NotImplementedError("Dose not support %s norm" % self.norm)

        return norm_id_replace_head, norm_id_replace_tail


class TransTry(PModel):  # rate:0.001 dimension:200,200 batch:4800 norm:L1 margin:2.0
    def __init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                 dimension_r, norm):

        PModel.__init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                        dimension_r, norm)

        bound = 6 / math.sqrt(self.dimension_e)

        with tf.device('/gpu'):
            self.embedding_entity = tf.get_variable('embedding_entity', [self.num_entity, self.dimension_r],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            self.embedding_relation = tf.get_variable('embedding_relation', [self.num_relation, self.dimension_r],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            self.parameter_relation = tf.get_variable('parameter_relation', [self.num_relation, 1],
                                                      initializer=tf.random_uniform_initializer(minval=0, maxval=0))
            self.parameter_head = tf.get_variable('parameter_head', [self.num_entity, 1],
                                                  initializer=tf.random_uniform_initializer(minval=0, maxval=0))
            self.parameter_tail = tf.get_variable('parameter_tail', [self.num_entity, 1],
                                                  initializer=tf.random_uniform_initializer(minval=0, maxval=0))
            self.variables.append(self.embedding_entity)
            self.variables.append(self.embedding_relation)
            self.variables.append(self.parameter_head)
            self.variables.append(self.parameter_tail)
            self.variables.append(self.parameter_relation)# 随机初始化向量
            print('finishing initializing')

    def train(self, inputs):
        with tf.device('/gpu'):
            embedding_relation = self.embedding_relation
            embedding_entity = self.embedding_entity
            parameter_h = self.parameter_head
            parameter_t = self.parameter_tail
            parameter_r = self.parameter_relation

            triple_positive, triple_negative = inputs  # triple_positive:(head_id,relation_id,tail_id)

            norm_entity = tf.nn.l2_normalize(embedding_entity, dim=1)  # 按行求L2标准化 shape=(14951, 100), dtype=float32
            norm_relation = tf.nn.l2_normalize(embedding_relation, dim=1)

            emb_p_h = tf.nn.embedding_lookup(norm_entity, triple_positive[:, 0])  # 取第0列的所有元素,即获取正样本中所有的头向量
            emb_p_t = tf.nn.embedding_lookup(norm_entity, triple_positive[:, 2])
            emb_p_r = tf.nn.embedding_lookup(norm_relation, triple_positive[:, 1])
            emb_p_ph = tf.nn.embedding_lookup(parameter_h, triple_positive[:, 0])
            emb_p_pt = tf.nn.embedding_lookup(parameter_t, triple_positive[:, 2])
            emb_p_pr = tf.nn.embedding_lookup(parameter_r, triple_positive[:, 1])

            emb_n_h = tf.nn.embedding_lookup(norm_entity, triple_negative[:, 0])
            emb_n_t = tf.nn.embedding_lookup(norm_entity, triple_negative[:, 2])
            emb_n_r = tf.nn.embedding_lookup(norm_relation, triple_negative[:, 1])
            emb_n_ph = tf.nn.embedding_lookup(parameter_h, triple_negative[:, 0])
            emb_n_pt = tf.nn.embedding_lookup(parameter_t, triple_negative[:, 2])
            emb_n_pr = tf.nn.embedding_lookup(parameter_r, triple_negative[:, 1])

            f_emb_p_h = tf.nn.l2_normalize(-emb_p_ph * emb_p_t * emb_p_r + emb_p_h, dim=1)
            f_emb_n_h = tf.nn.l2_normalize(-emb_n_ph * emb_n_t * emb_n_r + emb_n_h, dim=1)
            f_emb_p_t = tf.nn.l2_normalize(emb_p_pt * emb_p_h * emb_p_r + emb_p_t, dim=1)
            f_emb_n_t = tf.nn.l2_normalize(emb_n_pt * emb_n_h * emb_n_r + emb_n_t, dim=1)
            f_emb_p_r = tf.nn.l2_normalize(-emb_p_pr * emb_p_h * emb_p_t + emb_p_r, dim=1)
            f_emb_n_r = tf.nn.l2_normalize(-emb_n_pr * emb_n_h * emb_n_t + emb_n_r, dim=1)

            return PModel.get_loss(self, f_emb_p_h, f_emb_p_r, f_emb_p_t, f_emb_n_h, f_emb_n_r, f_emb_n_t)

    def test(self, inputs):
        with tf.device('/gpu'):
            embedding_relation = self.embedding_relation
            embedding_entity = self.embedding_entity
            parameter_h = self.parameter_head
            parameter_t = self.parameter_tail
            parameter_r = self.parameter_relation

            triple_test = inputs  # (headid, tailid, tailid)

            norm_emb_e = tf.nn.l2_normalize(embedding_entity, dim=1)
            norm_emb_r = tf.nn.l2_normalize(embedding_relation, dim=1)
            norm_vec_h = tf.nn.embedding_lookup(norm_emb_e, triple_test[0])
            norm_vec_t = tf.nn.embedding_lookup(norm_emb_e, triple_test[2])
            norm_vec_r = tf.nn.embedding_lookup(norm_emb_r, triple_test[1])
            norm_parameter_h = tf.nn.embedding_lookup(parameter_h, triple_test[0])
            norm_parameter_t = tf.nn.embedding_lookup(parameter_t, triple_test[2])
            norm_parameter_r = tf.nn.embedding_lookup(parameter_r, triple_test[1])

            f_norm_vec_h = tf.nn.l2_normalize(-norm_parameter_h * norm_vec_t * norm_vec_r + norm_vec_h)
            f_norm_vec_t = tf.nn.l2_normalize(norm_parameter_t * norm_vec_h * norm_vec_r + norm_vec_t)
            replace_emb_h = tf.nn.l2_normalize(-norm_parameter_h * norm_vec_t * norm_vec_r + norm_emb_e, dim=1)
            replace_emb_t = tf.nn.l2_normalize(norm_parameter_t * norm_vec_t * norm_vec_r + norm_emb_e, dim=1)
            replace_emb_r_h = tf.nn.l2_normalize(-norm_parameter_r * norm_emb_e * norm_vec_t + norm_vec_r, dim=1)
            replace_emb_r_t = tf.nn.l2_normalize(-norm_parameter_r * norm_vec_h * norm_emb_e + norm_vec_r, dim=1)

            if self.norm == 'L1':
                _, norm_id_replace_head = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(replace_emb_h + replace_emb_r_h - f_norm_vec_t), axis=1),
                    k=self.num_entity)
                _, norm_id_replace_tail = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(f_norm_vec_h + replace_emb_r_t - replace_emb_t), axis=1),
                    k=self.num_entity)
            elif self.norm == 'L2':
                _, norm_id_replace_head = tf.nn.top_k(
                    tf.reduce_sum((replace_emb_h + replace_emb_r_h - f_norm_vec_t) ** 2, axis=1),
                    k=self.num_entity)
                _, norm_id_replace_tail = tf.nn.top_k(
                    tf.reduce_sum((f_norm_vec_h + replace_emb_r_t - replace_emb_t) ** 2, axis=1),
                    k=self.num_entity)
            else:
                raise NotImplementedError("Dose not support %s norm" % self.norm)

            return norm_id_replace_head, norm_id_replace_tail


class TransBP(PModel):  # rate:0.001 dimension:200,200 batch:4800 norm:L1 margin:2.0
    def __init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                 dimension_r, norm):

        PModel.__init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                        dimension_r, norm)

        bound = 6 / math.sqrt(self.dimension_e)

        with tf.device('/gpu'):
            self.embedding_entity = tf.get_variable('embedding_entity', [self.num_entity, self.dimension_r],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            self.embedding_relation = tf.get_variable('embedding_relation', [self.num_relation, self.dimension_r],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            self.parameter_relation = tf.get_variable('parameter_relation', [self.num_relation, 1],
                                                      initializer=tf.random_uniform_initializer(minval=0, maxval=0))
            self.parameter_head = tf.get_variable('parameter_head', [self.num_entity, 1],
                                                  initializer=tf.random_uniform_initializer(minval=0, maxval=0))
            self.parameter_tail = tf.get_variable('parameter_tail', [self.num_entity, 1],
                                                  initializer=tf.random_uniform_initializer(minval=0, maxval=0))
            self.variables.append(self.embedding_entity)
            self.variables.append(self.embedding_relation)
            self.variables.append(self.parameter_head)
            self.variables.append(self.parameter_tail)
            self.variables.append(self.parameter_relation)# 随机初始化向量
            print('finishing initializing')

    def train(self, inputs):
        with tf.device('/gpu'):
            embedding_relation = self.embedding_relation
            embedding_entity = self.embedding_entity
            parameter_h = self.parameter_head
            parameter_t = self.parameter_tail
            parameter_r = self.parameter_relation

            triple_positive, triple_negative = inputs  # triple_positive:(head_id,relation_id,tail_id)

            norm_entity = tf.nn.l2_normalize(embedding_entity, dim=1)  # 按行求L2标准化 shape=(14951, 100), dtype=float32
            norm_relation = tf.nn.l2_normalize(embedding_relation, dim=1)

            emb_p_h = tf.nn.embedding_lookup(norm_entity, triple_positive[:, 0])  # 取第0列的所有元素,即获取正样本中所有的头向量
            emb_p_t = tf.nn.embedding_lookup(norm_entity, triple_positive[:, 2])
            emb_p_r = tf.nn.embedding_lookup(norm_relation, triple_positive[:, 1])
            emb_p_ph = tf.nn.embedding_lookup(parameter_h, triple_positive[:, 0])
            emb_p_pt = tf.nn.embedding_lookup(parameter_t, triple_positive[:, 2])
            emb_p_pr = tf.nn.embedding_lookup(parameter_r, triple_positive[:, 1])

            emb_n_h = tf.nn.embedding_lookup(norm_entity, triple_negative[:, 0])
            emb_n_t = tf.nn.embedding_lookup(norm_entity, triple_negative[:, 2])
            emb_n_r = tf.nn.embedding_lookup(norm_relation, triple_negative[:, 1])
            emb_n_ph = tf.nn.embedding_lookup(parameter_h, triple_negative[:, 0])
            emb_n_pt = tf.nn.embedding_lookup(parameter_t, triple_negative[:, 2])
            emb_n_pr = tf.nn.embedding_lookup(parameter_r, triple_negative[:, 1])

            f_emb_p_h = tf.nn.l2_normalize(-emb_p_ph * emb_p_t * emb_p_r + emb_p_h, dim=1)
            f_emb_n_h = tf.nn.l2_normalize(-emb_n_ph * emb_n_t * emb_n_r + emb_n_h, dim=1)
            f_emb_p_t = tf.nn.l2_normalize(emb_p_pt * emb_p_h * emb_p_r + emb_p_t, dim=1)
            f_emb_n_t = tf.nn.l2_normalize(emb_n_pt * emb_n_h * emb_n_r + emb_n_t, dim=1)
            f_emb_p_r = tf.nn.l2_normalize(emb_p_pr * emb_p_h * emb_p_t + emb_p_r, dim=1)
            f_emb_n_r = tf.nn.l2_normalize(emb_n_pr * emb_n_h * emb_n_t + emb_n_r, dim=1)

            return PModel.get_loss(self, f_emb_p_h, f_emb_p_r, f_emb_p_t, f_emb_n_h, f_emb_n_r, f_emb_n_t)

    def test(self, inputs):
        with tf.device('/gpu'):
            embedding_relation = self.embedding_relation
            embedding_entity = self.embedding_entity
            parameter_h = self.parameter_head
            parameter_t = self.parameter_tail
            parameter_r = self.parameter_relation

            triple_test = inputs  # (headid, tailid, tailid)

            norm_emb_e = tf.nn.l2_normalize(embedding_entity, dim=1)
            norm_emb_r = tf.nn.l2_normalize(embedding_relation, dim=1)
            norm_vec_h = tf.nn.embedding_lookup(norm_emb_e, triple_test[0])
            norm_vec_t = tf.nn.embedding_lookup(norm_emb_e, triple_test[2])
            norm_vec_r = tf.nn.embedding_lookup(norm_emb_r, triple_test[1])
            norm_parameter_h = tf.nn.embedding_lookup(parameter_h, triple_test[0])
            norm_parameter_t = tf.nn.embedding_lookup(parameter_t, triple_test[2])
            norm_parameter_r = tf.nn.embedding_lookup(parameter_r, triple_test[1])

            f_norm_vec_h = tf.nn.l2_normalize(-norm_parameter_h * norm_vec_t * norm_vec_r + norm_vec_h)
            f_norm_vec_t = tf.nn.l2_normalize(norm_parameter_t * norm_vec_h * norm_vec_r + norm_vec_t)
            replace_emb_h = tf.nn.l2_normalize(-norm_parameter_h * norm_vec_t * norm_vec_r + norm_emb_e, dim=1)
            replace_emb_t = tf.nn.l2_normalize(norm_parameter_t * norm_vec_t * norm_vec_r + norm_emb_e, dim=1)
            replace_emb_r_h = tf.nn.l2_normalize(norm_vec_r + norm_parameter_r * norm_emb_e * norm_vec_t, dim=1)
            replace_emb_r_t = tf.nn.l2_normalize(norm_vec_r + norm_parameter_r * norm_vec_h * norm_emb_e, dim=1)

            if self.norm == 'L1':
                _, norm_id_replace_head = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(replace_emb_h + replace_emb_r_h - f_norm_vec_t), axis=1),
                    k=self.num_entity)
                _, norm_id_replace_tail = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(f_norm_vec_h + replace_emb_r_t - replace_emb_t), axis=1),
                    k=self.num_entity)
            elif self.norm == 'L2':
                _, norm_id_replace_head = tf.nn.top_k(
                    tf.reduce_sum((replace_emb_h + replace_emb_r_h - f_norm_vec_t) ** 2, axis=1),
                    k=self.num_entity)
                _, norm_id_replace_tail = tf.nn.top_k(
                    tf.reduce_sum((f_norm_vec_h + replace_emb_r_t - replace_emb_t) ** 2, axis=1),
                    k=self.num_entity)
            else:
                raise NotImplementedError("Dose not support %s norm" % self.norm)

            return norm_id_replace_head, norm_id_replace_tail


class TransTry3(PModel):  # rate:0.001 dimension:200,200 batch:4800 norm:L1 margin:2.0
    def __init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                 dimension_r, norm):

        PModel.__init__(self, data_dir, negative_sampling, learning_rate, batch_size, max_iter, margin, dimension_e,
                        dimension_r, norm)

        bound = 6 / math.sqrt(self.dimension_e)

        with tf.device('/gpu'):
            self.embedding_entity = tf.get_variable('embedding_entity', [self.num_entity, self.dimension_r],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            self.embedding_relation = tf.get_variable('embedding_relation', [self.num_relation, self.dimension_r],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            self.parameter_relation = tf.get_variable('parameter_relation', [self.num_relation, 1],
                                                      initializer=tf.random_uniform_initializer(minval=0, maxval=0))
            self.parameter_head = tf.get_variable('parameter_head', [self.num_entity, 1],
                                                  initializer=tf.random_uniform_initializer(minval=0, maxval=0))
            self.parameter_tail = tf.get_variable('parameter_tail', [self.num_entity, 1],
                                                  initializer=tf.random_uniform_initializer(minval=0, maxval=0))
            self.variables.append(self.embedding_entity)
            self.variables.append(self.embedding_relation)
            self.variables.append(self.parameter_head)
            self.variables.append(self.parameter_tail)
            self.variables.append(self.parameter_relation)# 随机初始化向量
            print('finishing initializing')

    def train(self, inputs):
        with tf.device('/gpu'):
            embedding_relation = self.embedding_relation
            embedding_entity = self.embedding_entity
            parameter_h = self.parameter_head
            parameter_t = self.parameter_tail
            parameter_r = self.parameter_relation

            triple_positive, triple_negative = inputs  # triple_positive:(head_id,relation_id,tail_id)

            norm_entity = tf.nn.l2_normalize(embedding_entity, dim=1)  # 按行求L2标准化 shape=(14951, 100), dtype=float32
            norm_relation = tf.nn.l2_normalize(embedding_relation, dim=1)

            emb_p_h = tf.nn.embedding_lookup(norm_entity, triple_positive[:, 0])  # 取第0列的所有元素,即获取正样本中所有的头向量
            emb_p_t = tf.nn.embedding_lookup(norm_entity, triple_positive[:, 2])
            emb_p_r = tf.nn.embedding_lookup(norm_relation, triple_positive[:, 1])
            emb_p_ph = tf.nn.embedding_lookup(parameter_h, triple_positive[:, 0])
            emb_p_pt = tf.nn.embedding_lookup(parameter_t, triple_positive[:, 2])
            emb_p_pr = tf.nn.embedding_lookup(parameter_r, triple_positive[:, 1])

            emb_n_h = tf.nn.embedding_lookup(norm_entity, triple_negative[:, 0])
            emb_n_t = tf.nn.embedding_lookup(norm_entity, triple_negative[:, 2])
            emb_n_r = tf.nn.embedding_lookup(norm_relation, triple_negative[:, 1])
            emb_n_ph = tf.nn.embedding_lookup(parameter_h, triple_negative[:, 0])
            emb_n_pt = tf.nn.embedding_lookup(parameter_t, triple_negative[:, 2])
            emb_n_pr = tf.nn.embedding_lookup(parameter_r, triple_negative[:, 1])

            f_emb_p_h = tf.nn.l2_normalize(emb_p_ph * emb_p_t * emb_p_r + emb_p_h, dim=1)
            f_emb_n_h = tf.nn.l2_normalize(emb_n_ph * emb_n_t * emb_n_r + emb_n_h, dim=1)
            f_emb_p_t = tf.nn.l2_normalize(emb_p_pt * emb_p_h * emb_p_r + emb_p_t, dim=1)
            f_emb_n_t = tf.nn.l2_normalize(emb_n_pt * emb_n_h * emb_n_r + emb_n_t, dim=1)
            f_emb_p_r = tf.nn.l2_normalize(emb_p_pr * emb_p_h * emb_p_t + emb_p_r, dim=1)
            f_emb_n_r = tf.nn.l2_normalize(emb_n_pr * emb_n_h * emb_n_t + emb_n_r, dim=1)

            return PModel.get_loss(self, f_emb_p_h, f_emb_p_r, f_emb_p_t, f_emb_n_h, f_emb_n_r, f_emb_n_t)

    def test(self, inputs):
        with tf.device('/gpu'):
            embedding_relation = self.embedding_relation
            embedding_entity = self.embedding_entity
            parameter_h = self.parameter_head
            parameter_t = self.parameter_tail
            parameter_r = self.parameter_relation

            triple_test = inputs  # (headid, tailid, tailid)

            norm_emb_e = tf.nn.l2_normalize(embedding_entity, dim=1)
            norm_emb_r = tf.nn.l2_normalize(embedding_relation, dim=1)
            norm_vec_h = tf.nn.embedding_lookup(norm_emb_e, triple_test[0])
            norm_vec_t = tf.nn.embedding_lookup(norm_emb_e, triple_test[2])
            norm_vec_r = tf.nn.embedding_lookup(norm_emb_r, triple_test[1])
            norm_parameter_h = tf.nn.embedding_lookup(parameter_h, triple_test[0])
            norm_parameter_t = tf.nn.embedding_lookup(parameter_t, triple_test[2])
            norm_parameter_r = tf.nn.embedding_lookup(parameter_r, triple_test[1])

            f_norm_vec_h = tf.nn.l2_normalize(norm_parameter_h * norm_vec_t * norm_vec_r + norm_vec_h)
            f_norm_vec_t = tf.nn.l2_normalize(norm_parameter_t * norm_vec_h * norm_vec_r + norm_vec_t)
            replace_emb_h = tf.nn.l2_normalize(norm_parameter_h * norm_vec_t * norm_vec_r + norm_emb_e, dim=1)
            replace_emb_t = tf.nn.l2_normalize(norm_parameter_t * norm_vec_t * norm_vec_r + norm_emb_e, dim=1)
            replace_emb_r_h = tf.nn.l2_normalize(norm_vec_r + norm_parameter_r * norm_emb_e * norm_vec_t, dim=1)
            replace_emb_r_t = tf.nn.l2_normalize(norm_vec_r + norm_parameter_r * norm_vec_h * norm_emb_e, dim=1)

            if self.norm == 'L1':
                _, norm_id_replace_head = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(replace_emb_h + replace_emb_r_h - f_norm_vec_t), axis=1),
                    k=self.num_entity)
                _, norm_id_replace_tail = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(f_norm_vec_h + replace_emb_r_t - replace_emb_t), axis=1),
                    k=self.num_entity)
            elif self.norm == 'L2':
                _, norm_id_replace_head = tf.nn.top_k(
                    tf.reduce_sum((replace_emb_h + replace_emb_r_h - f_norm_vec_t) ** 2, axis=1),
                    k=self.num_entity)
                _, norm_id_replace_tail = tf.nn.top_k(
                    tf.reduce_sum((f_norm_vec_h + replace_emb_r_t - replace_emb_t) ** 2, axis=1),
                    k=self.num_entity)
            else:
                raise NotImplementedError("Dose not support %s norm" % self.norm)

            return norm_id_replace_head, norm_id_replace_tail
