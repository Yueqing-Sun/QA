#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: sunyueqing
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: sunyueqinghit@163.com
@File : answer_sentence_selection.py
@Time : 2019/5/21 16:02
@Site : 
@Software: PyCharm
'''
import random
import data_path
import json
import pickle
import time
from BM25 import BM25
import heapq
import jieba
import jieba.posseg as pseg
import distance
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from scipy.linalg import norm
# 引入gensim
from gensim.models.word2vec import LineSentence, Word2Vec

add_punc = ['[', ']', ':', '【', ' 】', '（', '）', '‘', '’', '{', '}', '⑦', '(', ')', '%', '^', '<', '>', '℃', '.', '-',
            '——', '—', '=', '&', '#', '@', '￥', '$']  # 定义要删除的特殊字符
stopwords = [line.strip() for line in open(data_path.stopwords_file, encoding='UTF-8').readlines()]
stopwords = stopwords + add_punc


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


def filter_stop(words):
    '''
    过滤停用词
    :param words:
    :return:
    '''
    return list(filter(lambda x: not ignore(x), words))


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


def train_test():
    with open(data_path.BM25Model, "rb") as f:
        # bm25 = BM25()
        bm25 = pickle.load(f)
    average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())

    # 读取未分词文件
    with open(data_path.passages_multi_sentences, encoding='utf-8') as fin:
        read_results = [json.loads(line.strip()) for line in fin.readlines()]

    # 读取训练文件
    # with open(data_path.answer_train, 'r', encoding='utf-8') as fin:  # 训练
    with open(data_path.answer_test, 'r', encoding='utf-8') as fin:  # 测试
        items = [json.loads(line.strip()) for line in fin.readlines()]

    pid_label = []
    pid_pre = []
    i = 0
    start1 = time.time()
    time1 = time.time()
    # 写入新的训练文件
    # with open(data_path.answer_sentence, 'w', encoding='utf-8')as fout:  # 训练
    with open(data_path.answer_sentence_test, 'w', encoding='utf-8')as fout:  # 测试
        for item in items:
            answer = item['answer_sentence']
            pid_label.append(item['pid'])  # 训练文件中的pid
            # scores = bm25.get_scores(dealwords(item['question']), average_idf)
            # # 取返回的第一个文档
            # idx = [scores.index(heapq.nlargest(1, scores)[0])]
            idx = [item['pid']]  # 直接用pid
            pid_pre.append(idx)
            i += 1
            if i % 100 == 0:
                data_path.logging.info("search {} done, use time {}s".format(i, time.time() - time1))
                print("search {} done, use time {}s".format(i, time.time() - time1))
                time1 = time.time()
            # 写入文件
            sentences = []
            for ii in idx:
                for id in read_results:
                    if ii == id['pid']:
                        for sen in id['document']:
                            if sen in answer:
                                continue
                            sentences.append(sen)

            data = {
                'question': item['question'],
                'right_answer': answer,
                'wrong_answer': sentences,
                'qid': item['qid']
            }
            fout.write(json.dumps(data, ensure_ascii=False))
            fout.write("\n")

    end1 = time.time()
    data_path.logging.info("Search {}, use time {}s".format(i, end1 - start1))


def notional_word(sentence):
    # TODO 换成ltp
    words = pseg.cut(sentence)
    flags = ['ns', 'a', 'v', 'n', 't', 'r', 'm', 'q']
    count = 0
    for w in words:
        # print(w.flag, w.word)
        if w.flag in flags:
            count += 1
    return count


def unigram(s1, s2):
    '''

    :param s1: 问句
    :param s2: 答案句
    :return:
    '''
    c1 = set(dealwords(s1))
    c2 = set(dealwords(s2))
    # print(c1&c2)
    return '%.4f' % (len(c1 & c2) / len(c2))


def bigram(s1, s2):
    list1 = dealwords(s1)
    list11 = []
    list2 = dealwords(s2)
    list22 = []
    for i in range(len(list1)):
        if i + 1 < len(list1):
            list11.append(list1[i] + list1[i + 1])
    for i in range(len(list2)):
        if i + 1 < len(list2):
            list22.append(list2[i] + list2[i + 1])
    c1 = set(list11)
    c2 = set(list22)
    # print(c1&c2)
    return '%.4f' % (len(c1 & c2) / len(list2))


def find_lcseque(s1, s2):
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    # d用来记录转移方向
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    s = []
    while m[p1][p2]:  # 不为None时
        c = d[p1][p2]
        if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  # 根据标记，向左找下一个
            p2 -= 1
        if c == 'up':  # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    return len(''.join(s))


def extractFeature():
    # 读取训练文件
    # with open(data_path.answer_sentence, 'r', encoding='utf-8') as fin:  # 训练
    with open(data_path.answer_sentence_test, 'r', encoding='utf-8') as fin:  # 测试
        train_data = [json.loads(line.strip()) for line in fin.readlines()]
    # 对qid进行排序
    train_data.sort(key=lambda x: x['qid'], reverse=False)
    # 提取特征，包括：1答案句词数，2答案句实词数，3词数差异，4unigram 词共现比例，5bigram 词共现比例，6最长公共子序列，7编辑距离，8余弦相似度，9embedding 语义相似度
    # TODO 9
    # with open(data_path.svm_rank_feature_train, 'w', encoding='utf-8')as fout:  # 训练
    with open(data_path.svm_rank_feature_test, 'w', encoding='utf-8')as fout:  # 测试
        for perQ in train_data:
            qid = perQ['qid']  # 问题编号
            question = perQ['question']
            right_answer = "".join(perQ['right_answer'])
            # 处理正确答案的特征
            right_answer_words = len(dealwords(right_answer))  # 1答案句词数
            rignt_notional_word = notional_word(right_answer)  # 2答案句实词数
            differ = len(dealwords(right_answer)) - len(dealwords(question))  # 3词数差异
            right_unigram = unigram(question, right_answer)  # 4unigram 词共现比例
            right_bigram = bigram(question, right_answer)  # 5bigram 词共现比例
            lcs = find_lcseque(question, right_answer)  # 6最长公共子序列
            edit = edit_distance(question, right_answer)  # 7编辑距离
            # a1, a2, a3 = similarity(question, right_answer)  # 余弦相似度
            fout.write(
                '3 ' + 'qid:' + str(qid) + ' 1:' + str(right_answer_words) +
                ' 2:' + str(rignt_notional_word) + ' 3:' + str(differ) +
                ' 4:' + str(right_unigram) + ' 5:' + str(right_bigram) +
                ' 6:' + str(lcs) + ' 7:' + str(edit) + "\n")

            # 处理错误答案的特征
            wrong = perQ['wrong_answer']
            for item in wrong:  # 对于每个错误答案
                if ignore(item) or len(dealwords(item)) < 2:
                    continue
                f1 = len(dealwords(item))
                f2 = notional_word(item)
                f3 = len(dealwords(item)) - len(dealwords(question))
                f4 = unigram(question, item)
                f5 = bigram(question, item)
                f6 = find_lcseque(question, item)
                f7 = edit_distance(question, item)
                # a1, a2, a3 = similarity(question, item)
                a = random.uniform(1, -3)
                fout.write('0 ' + 'qid:' + str(qid) + ' 1:' + str(f1) +
                           ' 2:' + str(f2) + ' 3:' + str(f3) +
                           ' 4:' + str(f4) + ' 5:' + str(f5) +
                           ' 6:' + str(f6) + ' 7:' + str(f7) + "\n")


def eval():
    predictions = [line.strip() for line in open(data_path.prediction, encoding='UTF-8').readlines()]
    with open(data_path.svm_rank_feature_test, encoding='utf-8') as fin:
        test = [line.strip() for line in fin.readlines()]
    right_index = []
    i = 0
    for item in test:
        list1 = item.split(" ")
        if list1[0] == '3':  # 根据赋值可改
            right_index.append(i)
        i += 1
    count = 0
    mrr = 0
    for j in range(len(right_index) - 1):
        list2 = predictions[right_index[j]:right_index[j + 1]]
        mrr += 1 / (list2.index(max(list2)) + 1)
        if list2.index(max(list2)) == 0:
            count += 1
    list2 = predictions[right_index[len(right_index) - 1]:len(predictions)]
    mrr += 1 / (list2.index(max(list2)) + 1)
    if list2.index(max(list2)) == 0:
        count += 1
    print("P: ", count / len(right_index))
    print("MRR: ", mrr / len(right_index))


def edit_distance(s1, s2):
    '''
    编辑距离
    :param s1:
    :param s2:
    :return:
    '''
    return distance.levenshtein(s1, s2)


def jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator


def tf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


def tfidf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


# 计算内积（加权）、余弦、Jaccard
def similarity(seg_list_1, seg_list_2):
    seg_1 = {}
    seg_2 = {}

    # 统计第一句话中每个词出现的数量
    for word in seg_list_1:
        try:
            seg_1[word] += 1.0
        except:
            seg_1[word] = 1.0

    # 统计第二句话中每个词出现的数量
    for word in seg_list_2:
        try:
            seg_2[word] += 1.0
        except:
            seg_2[word] = 1.0

    # 计算内积（加权）
    result_transvection_2 = 0.0
    # 计算向量模长
    cos_denominator_1 = 0.0
    cos_denominator_2 = 0.0
    for key in seg_1:
        try:
            result_transvection_2 += (seg_1[key] * seg_2[key])
        except:
            pass
        cos_denominator_1 += seg_1[key] ** 2

    for key in seg_2:
        cos_denominator_2 += seg_2[key] ** 2
    # 计算余弦
    result_cos = result_transvection_2 / (cos_denominator_1 * cos_denominator_2) ** 0.5
    # 计算Jaccard
    result_jaccard = result_transvection_2 / (cos_denominator_1 + cos_denominator_2 - result_transvection_2)
    return result_transvection_2, '%.4f' % result_cos, result_jaccard


def select_sentence():
    # 读取候选答案句文件
    with open(data_path.answer_sentence, encoding='utf-8') as fin:
        read_results = [json.loads(line.strip()) for line in fin.readlines()]
    i = 0
    start1 = time.time()
    time1 = time.time()
    with open(data_path.select_sentence, 'w', encoding='utf-8')as fout:
        for result in read_results:
            sentences = result['answer_sentence']
            eval = {}
            for sen in sentences:
                result_transvection_2, result_cos, result_jaccard = similarity(result['question'], sen)
                eval[sen] = result_cos
                # eval[sen] = edit_distance(result['question'], sen)
            ans = max(eval, key=eval.get)
            data = {
                'question': result['question'],
                'answer_sentence': ans
            }
            fout.write(json.dumps(data, ensure_ascii=False))
            fout.write("\n")
            i += 1
            if i % 100 == 0:
                data_path.logging.info("search {} done, use time {}s".format(i, time.time() - time1))
                print("search {} done, use time {}s".format(i, time.time() - time1))
                time1 = time.time()


def evaluate():
    # 读取答案句文件
    with open(data_path.select_sentence, encoding='utf-8') as fin:
        read_results = [json.loads(line.strip()) for line in fin.readlines()]

    # 读取测试集文件
    with open(data_path.train, encoding='utf-8') as fin:
        real_results = [json.loads(line.strip()) for line in fin.readlines()]
    count = 0
    for item1 in read_results:
        for item2 in real_results:
            if item1['question'] == item2['question']:
                if item1['answer_sentence'] == "".join(item2['answer_sentence']):
                    count += 1
                    break
                else:
                    print("**************")
                    print(item1['question'])
                    print(item1['answer_sentence'])
                    print("".join(item2['answer_sentence']))
                    break

    print(count / 5352)


def segment():
    '''
    提取词典
    :return:
    '''
    corpus = []
    # 读取未分词文件
    with open(data_path.passages_multi_sentences, encoding='utf-8') as fin:
        read_results = [json.loads(line.strip()) for line in fin.readlines()]
    for result in read_results:
        result['document'] = [' '.join(dealwords(sent)) for sent in result['document']]
        for item in result['document']:
            temp = item.split(" ")
            sen = []
            for i in temp:
                if not ignore(i):
                    sen.append(i)
            corpus.append(sen)
    print("分词结束，开始写入文件...")
    # 写回分词后的文件
    with open(data_path.sentences, 'w', encoding='utf-8') as fout:
        for item in corpus:
            for i in item:
                fout.write(i + " ")
            fout.write("\n")


def word2vec():
    # 模型训练
    sentences = LineSentence(data_path.sentences)
    model = Word2Vec(sentences, min_count=1, iter=1000)
    model.save(data_path.w2v)


def test_w2v():
    target = data_path.sentences  # 语料
    model = data_path.w2v  # word2vec模型
    model_w2v = Word2Vec.load(model)
    candidates = []
    with open(target, encoding='utf-8')as f:
        for line in f:
            candidates.append(line.strip().split())  # 将语料放到列表中便于操作
    text = "盐酸丁二胍什么时候被当做降糖药？"  # 待匹配文本
    words = list(jieba.cut(text.strip()))  # 分词
    flag = False
    word = []
    for w in words:
        if w not in model_w2v.wv.vocab:
            print("input word %s not in dict. skip this turn" % w)
        else:
            word.append(w)
    # 文本匹配
    res = []
    index = 0
    for candidate in candidates:
        # print("candidate", candidate)
        for c in candidate:
            if c not in model_w2v.wv.vocab:
                print("candidate word %s not in dict. skip this turn" % c)
                flag = True
        if flag:
            break
        score = model_w2v.n_similarity(word, candidate)
        resultInfo = {'id': index, "score": score, "text": " ".join(candidate)}
        res.append(resultInfo)
        index += 1
    res.sort(key=lambda x: x['score'], reverse=True)  # 进行排序

    # k = 0
    result = []  # 存放最终结果
    for i in range(len(res)):
        if res[i]['score'] > 0.80:  # 认为文本相似
            dict_temp = {res[i]['id']: res[i]['text'], "score": res[i]['score']}
            result.append(dict_temp)
    print(result)


if __name__ == '__main__':
    # select_sentence()
    # evaluate()
    # segment()
    # word2vec()
    # test_w2v()
    # train_test()
    # extractFeature()
    eval()
