#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: sunyueqing
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: sunyueqinghit@163.com
@File : data_path.py
@Time : 2019/5/9 18:47
@Site : 
@Software: PyCharm
'''

import logging
from os import path

# 3.1 原始数据
passages_multi_sentences = 'data/passages_multi_sentences.json'
new_test = 'data/new_test.json'
train = 'data/train.json'

# 停用词表
stopwords_file = 'data/stopwords(new).txt'

# 3.1 生成数据
passages_segment = 'preprocessed/passages_seg.json'
index_dir = 'preprocessed/index'
corpus = 'preprocessed/corpus.txt'
BM25Model = 'preprocessed/bm25.pkl'

# 3.2 原始数据
train_questions = 'question_classification/train_questions.txt'
test_questions = 'question_classification/test_questions.txt'

# 3.2 生成数据
train_questions_seg = 'question_classification/train_questions_seg.txt'
test_questions_seg = 'question_classification/test_questions_seg.txt'
trainFeature = 'question_classification/trainFeature.txt'
testFeature = 'question_classification/testFeature.txt'
trainFeature_little = 'question_classification/trainFeature_little.txt'
testFeature_little = 'question_classification/testFeature_little.txt'
question_corpus = 'question_classification/question_corpus.txt'
svm_model = 'question_classification/svm_model.model'

postags = 'question_classification/postags.txt'
parser = 'question_classification/parser.txt'


# 3.3 生成数据
answer_sentence = 'answer_sentence_selection/answer_sentence.json'
answer_sentence_test = 'answer_sentence_selection/answer_sentence_test.json'
select_sentence = 'answer_sentence_selection/select_sentence.json'
sentences = 'answer_sentence_selection/sentences.txt'
w2v = 'answer_sentence_selection/word2vec.model'
answer_train = 'answer_sentence_selection/train.json'
answer_test = 'answer_sentence_selection/test.json'
svm_rank_feature_train = 'answer_sentence_selection/svm_rank_feature_train.txt'
svm_rank_feature_test = 'answer_sentence_selection/svm_rank_feature_test.txt'
prediction = 'answer_sentence_selection/predictions'

# 3.4 生成数据
train_output = 'answer_span_selection/train.output.txt'
test_output = 'answer_span_selection/test_output.txt'
user_dict = 'answer_span_selection/user_dict.txt'
test_answer='answer_span_selection/test_answer.json'

# 日志记录
log_file = 'log.txt'
logging.basicConfig(filename=log_file, level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', \
                    datefmt='%a, %d %b %Y %H:%M:%S')
