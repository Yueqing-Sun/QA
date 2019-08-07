#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: sunyueqing
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: sunyueqinghit@163.com
@File : featureExtract.py
@Time : 2019/5/14 11:13
@Site :
@Software: PyCharm
'''
import jieba
import data_path
import os
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer
from svmutil import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV

add_punc = ['[', ']', ':', '【', ' 】', '（', '）', '‘', '’', '{', '}', '⑦', '(', ')', '%', '^', '<', '>', '℃', '.', '-',
            '——', '—', '=', '&', '#', '@', '￥', '$']  # 定义要删除的特殊字符
stopwords = [line.strip() for line in open(data_path.stopwords_file, encoding='UTF-8').readlines()]
stopwords = stopwords + add_punc
file_path = os.path.dirname(__file__)
os.chdir(file_path + 'lib\libsvm-3.23\python')


# 定义停止词
def ignore(s):
    # nbsp是空格的意思
    return s == 'nbsp' or s == ' ' or s == ' ' or s == '/t' or s == '/n' \
           or s == '，' or s == '。' or s == '！' or s == '、' or s == '―' \
           or s == '？' or s == '＠' or s == '：' \
           or s == '＃' or s == '%' or s == '＆' \
           or s == '（' or s == '）' or s == '《' or s == '》' \
           or s == '［' or s == '］' or s == '｛' or s == '｝' \
           or s == '*' or s == ',' or s == '.' or s == '&' \
           or s == '!' or s == '?' or s == ':' or s == ';' \
           or s == '-' or s == '&' \
           or s == '<' or s == '>' or s == '(' or s == ')' \
           or s == '[' or s == ']' or s == '{' or s == '}' or s == 'nbsp10' or s == '3.6' or s == 'about' or s == 'there' \
           or s == "see" or s == "can" or s == "U" or s == "L" or s == " " or s == "in" or s == ";" or s == "a" or s == "0144" \
           or s == "\n" or s == "our"


class LtpLanguageAnalysis(object):
    def __init__(self, model_dir="D:/ltp_data_v3.4.0"):
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(model_dir, "cws.model"))
        self.postagger = Postagger()
        self.postagger.load(os.path.join(model_dir, "pos.model"))
        self.parser = Parser()
        self.parser.load(os.path.join(model_dir, "parser.model"))
        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(model_dir, "ner.model"))  # 加载命名实体识别模型

    def analyze(self, text):
        # 分词
        words = self.segmentor.segment(text)
        print('\t'.join(words))

    def postags(self, words):
        # 词性标注
        postags = self.postagger.postag(words)
        # print('\t'.join(postags))
        return list(postags)
        # return '\t'.join(postags)

    def parse(self, words, postags):
        # 句法分析
        arcs = self.parser.parse(words, postags)
        # print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
        # print("\t".join(arc.relation for arc in arcs))
        return "\t".join(arc.relation for arc in arcs)

    def ner(self, words, postags):
        # 命名实体
        netag = self.recognizer.recognize(words, postags)
        for word, ntag in zip(words, netag):
            if ntag != 'O':
                print(word + '/' + ntag)
        print("\t".join(netag))

    def release_model(self):
        # 释放模型
        self.segmentor.release()
        self.postagger.release()
        self.parser.release()


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
    # words = filter_stop(words)  # 过滤没意义的词
    return words


def getcorpus():
    '''
    构建词袋模型
    :return:
    '''
    corpus = set()
    with open(data_path.train_questions, encoding='utf-8') as fin:
        read_results = [line.split("\t") for line in fin.readlines()]
        # print(read_results)
        # print(read_results[1][1])
    for result in read_results:
        seg = dealwords(result[1])
        for i in seg:
            if ignore(i) or i == "　" or i == " " or i == "\n" or i == "\t":
                continue
            corpus.add(i)
    with open(data_path.question_corpus, 'w', encoding='utf-8') as fout:
        j = 1
        for item in corpus:
            fout.write(item + "\t" + str(j) + "\n")
            j = j + 1


def getlittleLabel():
    '''
    得到小类的所有标签
    :return:
    '''
    with open(data_path.train_questions, encoding='utf-8') as fin:
        read_results = [line.split("\t") for line in fin.readlines()]
    labels = {}
    j = 1
    for item in read_results:
        if item[0] not in labels.keys():
            # print(j)
            labels[item[0]] = j
            j += 1
    # print(labels)
    labels['OBJ_ADDRESS'] = 85
    return labels


def extractFeature():
    '''
    提取特征,注释掉大类提取小类，注释掉小类提取大类
    :return:
    '''
    ltp = LtpLanguageAnalysis()
    # 大类
    label = {'DES': 1, 'HUM': 2, 'LOC': 3, 'NUM': 4, 'OBJ': 5, "TIME": 6, 'UNKNOWN': 7}
    # 小类
    # label = getlittleLabel()

    # 词袋特征
    with open(data_path.question_corpus, 'r', encoding='utf-8') as fin:  # question_corpus
        corpus = [line.strip().split("\t") for line in fin.readlines()]
    corpusDict = dict(corpus)

    # 词性特征
    with open(data_path.postags, 'r', encoding='utf-8') as fin:
        postags = [line.strip().split() for line in fin.readlines()]
    postagsDict = dict(postags)
    for key in postagsDict.keys():
        postagsDict[key] = int(postagsDict[key]) + 1143  # 7675

    # 句法特征
    with open(data_path.parser, 'r', encoding='utf-8') as fin:
        parsers = [line.strip().split() for line in fin.readlines()]
    parserDict = dict(parsers)
    for key in parserDict.keys():
        parserDict[key] = int(parserDict[key]) + 1172  # 7704

    # 读训练问题集 or 测试问题集
    with open(data_path.train_questions, encoding='utf-8') as fin:  #
        read_results = [line.split("\t") for line in fin.readlines()]

    # 开始提取特征
    with open(data_path.trainFeature, 'w', encoding='utf-8') as fout:  #
        for result in read_results:
            featureDict = {}
            # 大类
            temp = (result[0]).split("_")[0]
            # 小类
            # temp = result[0]
            if temp not in label.keys():
                print(temp)
                continue
            # 写入标签
            fout.write(str(label.get(temp)))
            fout.write("\t")
            # 写入词袋
            seg = dealwords(result[1])
            for i in seg:
                if i in corpusDict.keys():
                    featureDict[int(corpusDict.get(i))] = 1
                    # fout.write(corpusDict.get(i) + ":" + "1" + "\t")

            # # 写入词性
            # q_postags = ltp.postags(seg)
            # for i in q_postags:
            #     if i not in postagsDict.keys():
            #         print(i)
            #         continue
            #     featureDict[int(postagsDict.get(i))] = 1
            #     # fout.write(str(postagsDict.get(i)) + ":" + "1" + "\t")
            #
            # # 写入句法
            # q_parser = ltp.parse(seg, q_postags)
            # for i in q_parser.strip().split("\t"):
            #     if i not in parserDict.keys():
            #         print(i)
            #         continue
            #     featureDict[int(parserDict.get(i))] = 1
            #     # fout.write(str(parserDict.get(i)) + ":" + "1" + "\t")

            # feature id 要求排序
            for fea in sorted(featureDict.keys()):
                fout.write(str(fea) + ":" + "1" + "\t")
            fout.write("\n")
    print("Feature Extraction Complete")


def train_predict():
    '''
    svmtuil.py中含有下列主要函数
    svm_train() : 训练SVM模型
    svm_predict() : 预测测试数据结果
    svm_read_problem() : 读取数据.
    svm_load_model() : 加载SVM模型
    svm_save_model() :保存SVM模型.
    evaluations() : 检验预测结果.
    还有下列函数
    svm_problem(y, x)：返回一个problem类，作用等同于记录y，x列表
    svm_parameter('training_options'):返回一个parameter类，作用是记录参数选择
    '''
    # 大类训练
    y, x = svm_read_problem('D:/Work/IR/Lab2/question_classification/trainFeature.txt')
    yt, xt = svm_read_problem('D:/Work/IR/Lab2/question_classification/testFeature.txt')
    # 小类训练
    # y, x = svm_read_problem('D:/Work/IR/Lab2/question_classification/trainFeature_little.txt')
    # yt, xt = svm_read_problem('D:/Work/IR/Lab2/question_classification/testFeature_little.txt')

    prob = svm_problem(y, x)
    # -t 核函数类型：核函数设置类型(默认2)–RBF函数：exp(-gamma|u-v|^2)
    # -c cost：设置C-SVC，e -SVR和v-SVR的参数(损失函数)(默认1)
    # 对于RBF核函数，有一个参数。-g用来设置核函数中的gamma参数设置，也就是公式中的第一个r(gamma)，默认值是1/k（k是类别数）
    # -b 概率估计：是否计算SVC或SVR的概率估计，可选值0 或1，默认0；
    # g c 很重要
    param = svm_parameter('-t 2 -c 128 -g 0.0078125')  # 大类
    # param = svm_parameter('-t 2 -c 2048 -g 0.001953125')  # 小类
    # param = svm_parameter('-t 0 -c 2')
    model = svm_train(prob, param)
    svm_save_model(data_path.svm_model, model)

    # 预测
    # model=svm_load_model(data_path.svm_model)
    p_labs, p_acc, p_vals = svm_predict(yt, xt, model)
    ACC, MSE, SCC = evaluations(yt, p_labs)
    data_path.logging.info(p_labs)
    data_path.logging.info(p_acc)


# *****************逻辑回归********************#


def questions_seg():
    '''
    使用 jie ba 进行分词
    :return:
    '''
    # 啊啊啊...文件夹被我移来移去失效了
    file = [('D:/Work/IR/Lab2/question_classification/train_questions.txt',
             'D:/Work/IR/Lab2/question_classification/train_questions_seg.txt'),
            ('D:/Work/IR/Lab2/question_classification/test_questions.txt',
             'D:/Work/IR/Lab2/question_classification/test_questions_seg.txt')]
    for i in range(2):
        result = []
        with open(file[i][0], 'r', encoding='utf-8') as f:
            for line in f.readlines():
                attr = line.strip().split('\t')
                result.append("{}\t{}\n".format(attr[0], ' '.join(jieba.cut(attr[1]))))
        with open(file[i][1], 'w', encoding='utf-8') as f:
            f.writelines(result)


def load_data(size='fine'):
    '''
    加载数据
    :param size: 细粒度还是粗粒度，fine 细粒度（默认），rough（粗粒度）
    :return:
    '''
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    with open('D:/Work/IR/Lab2/question_classification/train_questions_seg.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            attr = line.strip().split('\t')
            x_train.append(attr[1])
            if size == 'rough':
                y_train.append(attr[0].split('_')[0])
            else:
                y_train.append(attr[0])
    with open('D:/Work/IR/Lab2/question_classification/test_questions_seg.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            attr = line.strip().split('\t')
            x_test.append(attr[1])
            if size == 'rough':
                y_test.append(attr[0].split('_')[0])
            else:
                y_test.append(attr[0])
    return x_train, y_train, x_test, y_test


def train_LogisticRegression():
    """使用逻辑回归进行分类
    CountVectorizer: 小类: 0.782509505703422  大类: 0.9019011406844106
    TfidfVectorizer: 小类: 0.7908745247148289  大类: 0.908745247148289
    """
    x_train, y_train, x_test, y_test = load_data(size='rough')
    tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    # tv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = tv.fit_transform(x_train)  # fit与transform的结合，先fit后transform
    test_data = tv.transform(x_test)

    lr = LogisticRegression(C=5000, solver='liblinear',
                            multi_class='ovr')  # penalty='l1',dual=False,max_iter=110,
    lr.fit(train_data, y_train)  # 拟合模型，用来训练LR分类器，其中X是训练样本，y是对应的标记向量
    print(lr.score(test_data, y_test))


def grid_search_lr():
    """ 网格搜索逻辑回归最优参数 """
    lr = LogisticRegression(solver='liblinear',multi_class='ovr')
    parameters = [
        {
            'C': [1, 10, 50, 100, 500, 1000, 5000, 10000],
        }]
    x_train, y_train, x_test, y_test = load_data(size='rough')
    # tv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = tv.fit_transform(x_train)
    test_data = tv.transform(x_test)

    clf = GridSearchCV(lr, parameters, cv=10, n_jobs=10)
    clf.fit(train_data, y_train)
    means = clf.cv_results_['mean_test_score']
    params = clf.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f  with:   %r" % (mean, param))
    print("%f  with:   %r" % (clf.best_score_, clf.best_params_))
    best_model = clf.best_estimator_
    print("test score: {}".format(best_model.score(test_data, y_test)))


if __name__ == '__main__':
    # getcorpus()
    # extractFeature()
    # train_predict()
    # questions_seg()
    # grid_search_lr()
    train_LogisticRegression()
