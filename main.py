import tensorflow as tf
import sys
import time
import argparse
import random
import numpy as np
import os.path
import math
import timeit
from multiprocessing import JoinableQueue, Queue, Process
from collections import defaultdict
from model import TransE, TransR, TransH, TransD, TransMS, TransEP, TransTry, TransTry3, TransBP


def train_operation(model, learning_rate=0.001, margin=1.0, optimizer_str='adam'):
    with tf.device('/gpu'):
        train_triple_positive_input = tf.placeholder(tf.int32, [None, 3])  # 行不定，列是3
        train_triple_negative_input = tf.placeholder(tf.int32, [None, 3])

        loss = model.train([train_triple_positive_input, train_triple_negative_input])

        if optimizer_str == 'gradient':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  # 梯度下降优化
        elif optimizer_str == 'rms':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)  # 自适应学习率优化
        elif optimizer_str == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # ADAM算法
        else:
            raise NotImplementedError("Dose not support %s optimizer" % optimizer_str)

        # grads = optimizer.compute_gradients(loss, model.variables)  # embedding_entity，embedding_relation
        # op_train = optimizer.apply_gradients(grads)
        op_train = optimizer.minimize(loss)
        return train_triple_positive_input, train_triple_negative_input, loss, op_train


def test_operation(model):
    with tf.device('/gpu'):
        test_triple = tf.placeholder(tf.int32, [3])
        norm_head_rank, norm_tail_rank = model.test(test_triple)
        return test_triple, norm_head_rank, norm_tail_rank


def test_link_prediction(model):
    with tf.device('/gpu'):
        norm_rank_head = []
        norm_rank_tail = []
        norm_filter_rank_head = []
        norm_filter_rank_tail = []

        testing_data = model.triple_test  # from test.txt
        hr_t = model.hr_t
        tr_h = model.tr_h
        n_test = args.n_test  # number of triples for test during the training
        if n_iter == args.max_iter - 1:
            n_test = model.num_triple_test
            print('the last link prediction is: ')
        test_idx = np.random.permutation(model.num_triple_test)
        for j in range(n_test):
            t = testing_data[test_idx[j]]
            norm_id_replace_head, norm_id_replace_tail = session.run([norm_head_rank, norm_tail_rank], {test_triple: t})

            norm_hrank = 0
            norm_fhrank = 0
            for i in range(len(norm_id_replace_head)):
                val = norm_id_replace_head[-i - 1]
                if val == t[0]:
                    break
                else:
                    norm_hrank += 1
                    norm_fhrank += 1
                    if val in tr_h[(t[2], t[1])]:
                        norm_fhrank -= 1
            norm_trank = 0
            norm_ftrank = 0
            for i in range(len(norm_id_replace_tail)):
                val = norm_id_replace_tail[-i - 1]
                if val == t[2]:
                    break
                else:
                    norm_trank += 1
                    norm_ftrank += 1
                    if val in hr_t[(t[0], t[1])]:
                        norm_ftrank -= 1

            norm_rank_head.append(norm_hrank)
            norm_rank_tail.append(norm_trank)
            norm_filter_rank_head.append(norm_fhrank)
            norm_filter_rank_tail.append(norm_ftrank)

        norm_mean_rank_head = np.sum(norm_rank_head, dtype=np.float32) / n_test
        norm_mean_rank_tail = np.sum(norm_rank_tail, dtype=np.float32) / n_test
        norm_filter_mean_rank_head = np.sum(norm_filter_rank_head, dtype=np.float32) / n_test
        norm_filter_mean_rank_tail = np.sum(norm_filter_rank_tail, dtype=np.float32) / n_test

        norm_hit10_head = np.sum(np.asarray(np.asarray(norm_rank_head) < 10, dtype=np.float32)) / n_test
        norm_hit10_tail = np.sum(np.asarray(np.asarray(norm_rank_tail) < 10, dtype=np.float32)) / n_test
        norm_filter_hit10_head = np.sum(
            np.asarray(np.asarray(norm_filter_rank_head) < 10, dtype=np.float32)) / n_test
        norm_filter_hit10_tail = np.sum(
            np.asarray(np.asarray(norm_filter_rank_tail) < 10, dtype=np.float32)) / n_test

        print('iter:%d --norm mean rank: %.2f --norm hit@10: %.2f%%' % (
            n_iter, (norm_mean_rank_head + norm_mean_rank_tail) / 2, (norm_hit10_tail + norm_hit10_head) / 2 * 100))
        print('iter:%d --norm filter mean rank: %.2f --norm filter hit@10: %.2f%%' % (
            n_iter, (norm_filter_mean_rank_head + norm_filter_mean_rank_tail) / 2,
            (norm_filter_hit10_tail + norm_filter_hit10_head) / 2 * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TransX")
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='the directory of dataset',
                        default='./FB15K/')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning rate', default=0.001)
    parser.add_argument('--batch_size', dest='batch_size', type=int, help="batch size", default=4800)
    parser.add_argument('--max_iter', dest='max_iter', type=int, help='maximum interation', default=200)
    parser.add_argument('--optimizer', dest='optimizer', type=str, help='optimizer', default='adam')
    parser.add_argument('--dimension_e', dest='dimension_e', type=int, help='entity dimension', default=200)
    parser.add_argument('--dimension_r', dest='dimension_r', type=int, help='relation dimension', default=200)
    parser.add_argument('--margin', dest='margin', type=float, help='margin', default=3.0)
    parser.add_argument('--norm', dest='norm', type=str, help='L1 or L2 norm', default='L2')
    parser.add_argument('--negative_sampling', dest='negative_sampling', type=str,
                        help='choose unif or bern to generate negative examples', default='unif')
    parser.add_argument('--evaluate_per_iteration', dest='evaluate_per_iteration', type=int,
                        help='evaluate the training result per x iteration', default=5)
    parser.add_argument('--n_test', dest='n_test', type=int, help='number of triples for test during the training',
                        default=300)
    args = parser.parse_args()
    print(args)

    model = TransBP(negative_sampling=args.negative_sampling, data_dir=args.data_dir,
                      learning_rate=args.learning_rate, batch_size=args.batch_size,
                      max_iter=args.max_iter, margin=args.margin,
                      dimension_e=args.dimension_e, dimension_r=args.dimension_r, norm=args.norm)

    train_triple_positive_input, train_triple_negative_input, loss, op_train = \
        train_operation(model, learning_rate=args.learning_rate, margin=args.margin, optimizer_str=args.optimizer)
    test_triple, norm_head_rank, norm_tail_rank = test_operation(model)
    with tf.Session() as session:
        tf.initialize_all_variables().run()

        for embeddings in model.variables:  # 对初始化的向量组L2标准化
            norm_emb = session.run(tf.nn.l2_normalize(embeddings, dim=1))
            session.run(tf.assign(embeddings, norm_emb))

        for n_iter in range(args.max_iter):
            accu_loss = 0.
            start_time = timeit.default_timer()
            prepare_time = 0.

            for tp, tn, t in model.training_data_batch(batch_size=args.batch_size):
                l, _ = session.run([loss, op_train], {train_triple_positive_input: tp, train_triple_negative_input: tn})
                accu_loss += l
                prepare_time += t
            print('iter[%d] ---loss: %.5f ---time: %.2f ---prepare time : %.2f' %
                  (n_iter, accu_loss, timeit.default_timer() - start_time, prepare_time))
            if n_iter % args.evaluate_per_iteration == 0 or n_iter == 0 or n_iter == args.max_iter - 1:
                test_link_prediction(model)



