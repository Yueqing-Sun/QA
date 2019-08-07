#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: sunyueqing
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: sunyueqinghit@163.com
@File : BM25.py
@Time : 2019/5/18 20:15
@Site :
@Software: PyCharm
'''

import math
from six import iteritems
from six.moves import xrange
import data_path
import jieba
import json
import jieba.posseg as pseg
import time
import pickle
import heapq

# BM25 parameters.
PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25

add_punc = ['[', ']', ':', '【', ' 】', '（', '）', '‘', '’', '{', '}', '⑦', '(', ')', '%', '^', '<', '>', '℃', '.', '-',
            '——', '—', '=', '&', '#', '@', '￥', '$']  # 定义要删除的特殊字符
stopwords = [line.strip() for line in open(data_path.stopwords_file, encoding='UTF-8').readlines()]
stopwords = stopwords + add_punc


class BM25(object):

    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = sum(map(lambda x: float(len(x)), corpus)) / self.corpus_size
        self.corpus = corpus
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {}  # 存储每个词及出现了该词的文档数量
        self.idf = {}  # 存储每个词的idf值
        self.initialize()

    def initialize(self):
        for document in self.corpus:
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, document, index, average_idf):
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON * average_idf
            score += (idf * self.f[index][word] * (PARAM_K1 + 1)
                      / (self.f[index][word] + PARAM_K1 * (
                            1 - PARAM_B + PARAM_B * len(self.corpus[index]) / self.avgdl)))
        return score

    def get_scores(self, document, average_idf):
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(document, index, average_idf)
            scores.append(score)
        return scores

    # 总共有N篇文档，传来的doc为查询文档，计算doc与所有文档匹配
    # 后的得分score，总共有多少篇文档，scores列表就有多少项，
    # 每一项为doc与这篇文档的得分，所以分清楚里面装的是文档得分，
    # 不是词语得分。
    def simall(self, doc):
        scores = []
        for index in range(self.corpus_size):
            score = self.get_scores(doc, index)
            scores.append(score)
        return scores

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


def get_bm25_weights(corpus):
    bm25 = BM25(corpus)
    average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())

    weights = []
    for doc in corpus:
        scores = bm25.get_scores(doc, average_idf)
        weights.append(scores)

    return weights


def filter_stop(words):
    '''
    过滤停用词
    :param words:
    :return:
    '''
    return list(filter(lambda x: x not in stopwords, words))


def dealwords(sent):
    '''
    处理句子
    :param self:
    :param sent:
    :return:
    '''
    words = list(jieba.cut(sent))  # 分词
    words = filter_stop(words)  # 过滤没意义的词
    return words


def segment():
    '''
    提取词典
    :return:
    '''
    corpus = set()
    # 读取未分词文件
    with open(data_path.passages_multi_sentences, encoding='utf-8') as fin:
        read_results = [json.loads(line.strip()) for line in fin.readlines()]
    for result in read_results:
        result['document'] = [' '.join(dealwords(sent)) for sent in result['document']]
        for item in result['document']:
            temp = item.split(" ")
            for i in temp:
                # print(i)
                if i not in stopwords:
                    corpus.add(i)
    print("分词结束，开始写入文件...")
    # 写回分词后的文件
    with open(data_path.corpus, 'w', encoding='utf-8') as fout:
        for item in corpus:
            fout.write(item + '\n')


def build_BM25Model():
    '''
    存储bm25模型,用时232s
    :return:
    '''
    docs = []  # 所有文档列表,词表示
    # 读取未分词文件
    with open(data_path.passages_multi_sentences, encoding='utf-8') as fin:
        read_results = [json.loads(line.strip()) for line in fin.readlines()]

    for result in read_results:
        words_in_document = []
        for sent in result['document']:
            for i in (dealwords(sent)):  # 分词
                words_in_document.append(i)
        docs.append(words_in_document)
        # print(words_in_document)
        print(len(docs))
    print("建立BM25模型...")
    print(len(docs))
    bm25Model = BM25(docs)
    bm25Model.save_model(data_path.BM25Model)


def search():
    with open(data_path.BM25Model, "rb") as f:
        bm25 = pickle.load(f)
    query = "我爱罗是《火影忍者》中谁的弟弟？？"
    # print(dealwords(query))
    average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())
    scores = bm25.get_scores(dealwords(query), average_idf)
    for i in heapq.nlargest(3, scores):
        idx = scores.index(i)
        print(idx)


def train_test():
    # 读取未分词文件
    with open(data_path.passages_multi_sentences, encoding='utf-8') as fin:
        read_results = [json.loads(line.strip()) for line in fin.readlines()]

    # 读取训练文件
    with open(data_path.train, 'r', encoding='utf-8') as fin:  # 训练
        # with open(data_path.answer_test, 'r', encoding='utf-8') as fin:  # 测试
        items = [json.loads(line.strip()) for line in fin.readlines()]

    i = 0
    start1 = time.time()
    time1 = time.time()
    count = 0
    mrr = 0
    for item in items:
        answer = item['answer_sentence']
        pid = item['pid']  # 直接用pid
        query = item['question']

        # 得到候选答案句
        docs = []
        for j in read_results:
            if pid == j['pid']:
                for sen in j['document']:
                    sentence = []
                    for word in dealwords(sen):
                        sentence.append(word)
                    docs.append(sentence)
        bm25 = BM25(docs)  # 使用文档中的句子建立BM25模型
        average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())
        scores = bm25.get_scores(dealwords(query), average_idf)
        # TODO
        rank = 0
        for ii in heapq.nlargest(3, scores):
            rank += 1
            idx = scores.index(ii)
            for an in answer:
                if docs[idx] == dealwords(an):
                    mrr += (1 / rank)
                    count += 1
                    print(an)
                    break

    print("P: ", count / len(items))
    print("MRR: ", mrr / len(items))
    end1 = time.time()
    data_path.logging.info("P {}, MRR {}s".format(count / len(items), mrr / len(items)))


def get_test():
    # 读取测试文件
    with open(data_path.new_test, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    i = 0
    start1 = time.time()
    time1 = time.time()
    with open(data_path.new_test, 'w', encoding='utf-8') as fout:
        for item in items:
            query = item['question']

            # 得到候选答案句
            docs = []
            docs2 = []
            for sen in item['document']:
                sentence = []
                for word in dealwords(sen):
                    sentence.append(word)
                docs.append(sentence)
                docs2.append(sen)

            bm25 = BM25(docs)  # 使用文档中的句子建立BM25模型
            average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())
            scores = bm25.get_scores(dealwords(query), average_idf)
            # s = []
            # for ii in heapq.nlargest(3, scores):
            #     idx = scores.index(ii)
            #     s.append(docs2[idx])
            # item['answer_sentence'] = ''.join(s)  # docs2[idx]
            item['answer_sentence'] = docs2[scores.index(max(scores))]

            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

            i += 1
            if i % 100 == 0:
                data_path.logging.info("查询 {} 完成, 用时 {}s".format(i, time.time() - time1))
                print("查询 {} 完成, 用时 {}s".format(i, time.time() - time1))
                time1 = time.time()

    end1 = time.time()
    data_path.logging.info("抽取答案句用时 {}s".format(end1 - start1))


if __name__ == '__main__':
    start = time.time()
    # segment()
    # build_BM25Model()
    # train_test()
    # search()
    get_test()
    end = time.time()
    print("查询用时： ", end - start)
