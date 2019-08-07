#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import data_path
import re
import json
import jieba
from nltk.translate.bleu_score import sentence_bleu
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer
import os
import metric

LTP_DATA_DIR = 'D:/ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
postagger = Postagger()
postagger.load(os.path.join(LTP_DATA_DIR, "pos.model"))

recognizer = NamedEntityRecognizer()
recognizer.load(os.path.join(LTP_DATA_DIR, "ner.model"))  # 加载命名实体识别模型


def dealwords(sent):
    '''
    处理句子
    :param self:
    :param sent:
    :return:
    '''
    words = list(jieba.cut(sent))  # 分词
    return words


def lcs(string1_list, string2_list):
    n = len(string1_list)
    m = len(string2_list)

    if m == 0 or n == 0:
        return -1
    c = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if string1_list[i - 1] == string2_list[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
            else:
                c[i][j] = max(c[i][j - 1], c[i - 1][j])
    return c[-1][-1]


def get_most_lcs_sentence(text, question):
    sentences = []
    start_idx = 0
    for i in range(len(text)):
        word = text[i]
        if word in ['，', '。', '！', '：', ':', '……', '？', ',', '?', '；', ';', '.', '《', '》', '、']:
            if i > start_idx:
                sentence = text[start_idx: i]
                sentences.append(sentence)
                start_idx = i + 1
        if i == len(text) - 1 and i >= start_idx:
            sentence = text[start_idx:]
            sentences.append(sentence)

    most_lcs_sentence = []
    most_lcs_length = 0
    for sen in sentences:
        lcs_lenght = lcs(sen, question)
        if lcs_lenght > most_lcs_length:
            most_lcs_length = lcs_lenght
            most_lcs_sentence = sen
    # print(most_lcs_sentence)
    return most_lcs_sentence


def extract_answer(sentence, question):
    que_words = ['都有哪些', '什么地方', '什么内容', '什么时候',
                 '哪三段', '为什么', '怎么样', '哪一部', '多少个', '多少钱', '哪个人', '什么样', '哪一年', '哪一天',
                 '什么', '哪里', '哪儿', '几个', '如何', '几层', '哪年', '多少', '怎么', '哪些',
                 '何时', '几条', '哪个', '多重', '多长', '多大', '多久', '多宽', '多深', '多远',
                 '哪天', '是谁', '时间', '第几', '谁', '哪', '几', '吗', '多']
    answer = []
    for qw in que_words:  # 对于问句中的每一个词
        if qw in question:  # 如果在疑问词列表中
            que_len = len(question)
            qw_index = question.index(qw)
            qw_b_index = qw_index - 1
            qw_a_index = qw_index + 1
            if qw_b_index < 0:
                # print("2")
                # 疑问词在开头
                while qw_a_index < que_len - 1 and question[qw_a_index] not in sentence:
                    qw_a_index += 1
                if question[qw_a_index] in sentence:
                    answer = sentence[:sentence.index(question[qw_a_index])]
                    if len(answer) == 0:
                        # 如果为空，取后面的
                        answer = sentence[sentence.index(question[qw_a_index]):]
                else:
                    answer = sentence[:qw_a_index]
            elif qw_a_index >= que_len:
                # print("1")
                # 疑问词在结尾
                while qw_b_index > 0 and question[qw_b_index] not in sentence:
                    qw_b_index -= 1
                if question[qw_b_index] in sentence:
                    # 如果匹配到了结尾，则取前面的所有词
                    if sentence.index(question[qw_b_index]) == len(sentence) - 1:
                        # print(sentence)
                        answer = sentence[0:sentence.index(question[qw_b_index])]
                    else:
                        answer = sentence[sentence.index(question[qw_b_index]) + 1:]
                else:
                    answer = sentence[qw_b_index:]
            else:
                # print("0")
                # 疑问词在中间
                # 找到匹配左边最右的下标
                # 找到匹配右边最左的下标
                while qw_a_index < que_len - 1 and question[qw_a_index] not in sentence:
                    qw_a_index += 1
                while qw_b_index > 0 and question[qw_b_index] not in sentence:
                    qw_b_index -= 1

                if question[qw_b_index] in sentence:
                    start_index = sentence.index(question[qw_b_index]) + 1
                    if start_index >= len(sentence) - 1:
                        start_index -= 1
                else:
                    start_index = 0  # qw_b_index
                # print(start_index)

                if question[qw_a_index] in sentence:
                    end_index = sentence.index(question[qw_a_index])
                    if end_index <= 0:
                        end_index += 1
                else:
                    end_index = len(sentence)  # qw_a_index
                # print(end_index)
                if end_index == 1:
                    answer = sentence[1:]
                elif start_index == end_index:
                    answer = sentence[end_index:]
                elif start_index > end_index:
                    answer = sentence[start_index:]
                else:
                    answer = sentence[start_index:end_index]
                    # print(answer)
            break  # 只考虑一个疑问词

            # TODO if start_index==end_index

            # if abs(qw_index - qw_b_index) <= 1:
            #     if question[qw_b_index] in sentence:
            #         answer = [].append(sentence[start_index])
            # elif 1 >= abs(qw_index - qw_a_index):
            #     if question[qw_a_index] in sentence:
            #         answer = [].append(sentence[end_index - 1])
    # if len(answer) > 0:
    #     print(answer)
    answer = time_question(answer, question, sentence)
    answer = hum_question(answer, question, sentence)
    answer = number_question(answer, question, sentence)

    if answer and answer[0] in ['：', ':', '，', '》']:  # 以这些符号开头，删去
        if len(answer) == 1:
            return []
        answer = answer[1:]
    if answer and answer[-1] in ['。', '《', ':', '，', '；', '、']:  # 以这些符号结尾，删去
        if len(answer) == 1:
            return []
        answer = answer[:len(answer) - 1]
    # 取冒号右侧
    if '：' in answer:
        return answer[answer.index("：") + 1:]
    if ':' in answer:
        return answer[answer.index(":") + 1:]
    if '，' in answer:  # 暴力去逗号
        return answer[0:answer.index("，")]
    if '。' in answer:  # 只取一句
        return answer[0:answer.index("。")]
    # if not answer or len(''.join(answer)) < 1 or answer is None:
    #     return sentence
    return answer


def time_question(answer, question, answer_sentence):
    sentence = ''.join(answer)
    que_words = ['什么时候', '哪一年', '哪一天', '哪年', '何时', '多久', '时间', '哪天']
    for qw in que_words:  # 对于问句中的每一个词
        if qw in question:  # 如果在疑问词列表中
            # 匹配answer中的时间
            # XX年
            # XX年XX月
            # XX年XX月XX日
            # XX-XX-XX
            mat = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日)", sentence)
            if mat:
                return [mat.group(0)]
            mat = re.search(r"(\d{4}年\d{1,2}月)", sentence)
            if mat:
                return [mat.group(0)]
            mat = re.search(r"(\d{4}年)", sentence)
            if mat:
                return [mat.group(0)]
            mat = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", sentence)
            if mat:
                return [mat.group(0)]
            # mat = re.search(r"([一二三四五六七八九零十百千万亿]+年)", ''.join(sentence))
            # if mat:
            #     return [mat.group(0)]

            mat = re.search(r"(\d{1,2}月\d{1,2}日)", sentence)
            if mat:
                return [mat.group(0)]

            mat = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日)", ''.join(answer_sentence))
            if mat:
                return [mat.group(0)]
            mat = re.search(r"(\d{4}年\d{1,2}月)", ''.join(answer_sentence))
            if mat:
                return [mat.group(0)]
            mat = re.search(r"(\d{4}年)", ''.join(answer_sentence))
            if mat:
                return [mat.group(0)]
            mat = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", ''.join(answer_sentence))
            if mat:
                return [mat.group(0)]
            # mat = re.search(r"([一二三四五六七八九零十百千万亿]+年)", ''.join(answer_sentence))
            # if mat:
            #     return [mat.group(0)]

    return answer


def hum_question(answer, question, answer_sentence):
    sentence = ''.join(answer)
    que_words = ['谁']
    for qw in que_words:  # 对于问句中的每一个词
        if qw in question:  # 如果在疑问词列表中
            # 匹配answer中的人名
            postags = postagger.postag(answer)
            netag = recognizer.recognize(answer, postags)
            SNh = []
            for word, ntag in zip(answer, netag):
                if ntag == 'S-Nh':
                    SNh.append(word)
            if len(SNh) > 0:
                return SNh
            else:
                # 如果答案中没有，去原句中找
                postags = postagger.postag(answer_sentence)
                netag = recognizer.recognize(answer_sentence, postags)
                SNh = []
                for word, ntag in zip(answer_sentence, netag):
                    if ntag == 'S-Nh':
                        SNh.append(word)
                if len(SNh) > 0:
                    return SNh
    return answer


def number_question(answer, question, answer_sentence):
    que_words = ['几', '几个', '几条', '几层', '第几']
    for qw in que_words:  # 对于问句中的每一个词
        if qw in question or ('多少' in question and '个' in question):  # 如果在疑问词列表中
            mat = re.search(r"(\d+)", ''.join(answer))
            if mat:
                return [mat.group(0)]
            mat = re.search(r"(([一二三四五六七八九零十百千万亿]+|[0-9]+[,]*[0-9]+.[0-9]+))", ''.join(answer))
            if mat:
                return [mat.group(0)]
            mat = re.search(r"(\d+)", ''.join(answer_sentence))
            if mat:
                return [mat.group(0)]
            mat = re.search(r"(([一二三四五六七八九零十百千万亿]+|[0-9]+[,]*[0-9]+.[0-9]+))", ''.join(answer_sentence))
            if mat:
                return [mat.group(0)]
    return answer


def read_stopwords():
    stopwords = codecs.open(data_path.stopwords_file, encoding='utf-8').read()
    stopwords = stopwords.split("\n")
    return stopwords


def read_data():
    # 读取未分词文件

    segmentor = Segmentor()  # 初始化实例
    segmentor.load_with_lexicon(cws_model_path, data_path.user_dict)

    with open(data_path.train, encoding='utf-8') as fin:
        read_results = [json.loads(line.strip()) for line in fin.readlines()]
    answer_sentence = []
    question = []
    ans = []
    for item in read_results:
        answer_sentence.append(list(segmentor.segment(''.join(item['answer_sentence']).strip())))
        # print(list(jieba.cut(''.join(item['answer_sentence']))))
        que = item['question']
        if que[-1] == '？':
            que = que[0:len(que) - 1]
        question.append(list(segmentor.segment(que.strip())))
        ans.append(item['answer'])

    segmentor.release()  # 释放模型
    assert len(answer_sentence) == len(question)
    return answer_sentence, question, ans


def bleu1(prediction, ground_truth):
    '''
    计算单个预测答案prediction和单个真实答案ground_truth之间的字符级别的bleu1值,(可能会有warning， 不用管)
    Args:
        prediction: 预测答案（未分词的字符串）
        ground_truth: 真实答案（未分词的字符串）
    Returns:
        floats of bleu1
    eg:
    >>> prediction = '北京天安门'
    >>> ground_truth = '天安门'
    >>> bleu1(prediction, ground_truth)
    >>> 0.6
    '''
    prediction = ' '.join(prediction).split()
    ground_truth = [' '.join(ground_truth).split()]
    bleu1 = sentence_bleu(ground_truth, prediction, weights=(1, 0, 0, 0))
    return bleu1


def main():
    # lcs_list = lcs(['1', '2', '3','3','3'], ['1','3','3'])
    # print(lcs_list)
    text, question, ans = read_data()
    stopwords = read_stopwords()
    most_lcs_sentences = []
    # for i in range(len(text)):
    #     most_lcs_sentence = get_most_lcs_sentence(text[i], question[i])
    #     most_lcs_sentences.append(most_lcs_sentence)
    # most_lcs_sentences = [[word for word in sen if (word not in stopwords)] for sen in most_lcs_sentences]
    # question = [[word for word in sen if (word not in stopwords)] for sen in question]
    #
    # assert len(most_lcs_sentences) == len(question)
    # for i in range(len(question)):
    #     print(i)
    #     print(most_lcs_sentences[i])
    #     print(question[i])

    answers = []
    for i in range(len(question)):
        answer = extract_answer(text[i], question[i])
        answers.append(answer)

    bleu = 0
    bleu2 = 0
    a = []
    b = []
    with open(data_path.train_output, 'w', encoding='utf-8') as f:
        for i in range(len(answers)):
            # if len(answers[i]) == 0:
            #     answers[i] = most_lcs_sentences[i]
            bleu += bleu1(''.join(answers[i]), ''.join(ans[i]))
            bleu2 += bleu1(''.join(text[i]), ''.join(ans[i]))
            f.write(
                '<qid_' + str(i) + '> ||| ' + ''.join(answers[i]) + ' ||| ' + ''.join(ans[i]) + '\n')
            a.append(''.join(answers[i]))
            b.append(''.join(ans[i]))
    print("bleu : ", bleu / len(question))
    # print("bleu2 : ", bleu2 / 5352)
    print("exact_match: ", metric.exact_match(a, b))
    print("p, r, f1: ", metric.precision_recall_f1(a, b))


def get_test():
    segmentor = Segmentor()  # 初始化实例
    segmentor.load_with_lexicon(cws_model_path, data_path.user_dict)
    with open(data_path.new_test, encoding='utf-8') as fin:
        read_results = [json.loads(line.strip()) for line in fin.readlines()]
    answer_sentence = []
    question = []
    # ans = []
    for item in read_results:
        answer_sentence.append(list(segmentor.segment(''.join(item['answer_sentence']).strip())))
        que = item['question']
        if que[-1] == '？':
            que = que[0:len(que) - 1]
        question.append(list(segmentor.segment(que.strip())))
        # ans.append(item['answer'])

    segmentor.release()  # 释放模型

    answers = []
    for i in range(len(question)):
        answer = extract_answer(answer_sentence[i], question[i])
        answers.append(answer)
    j = 0
    with open(data_path.test_answer, 'w', encoding='utf-8') as f:
        for item in read_results:
            data = {
                'qid': item['qid'],
                'question': item['question'],
                'answer': ''.join(answers[j])
            }
            json_str = json.dumps(data, ensure_ascii=False)
            f.write(json_str)
            f.write("\n")
            j += 1

    with open(data_path.test_output, 'w', encoding='utf-8') as f:
        for i in range(len(answers)):
            f.write(
                '<qid_' + str(i) + '> ||| ' + ''.join(question[i]) + ' ||| ' + ''.join(answers[i]) + ' ||| ' + ''.join(
                    answer_sentence[i]) + '\n')


if __name__ == '__main__':
    # main()
    get_test()
    # print(time_question(
    #     ['1825年', '1月', '13日', '，', '县政府', '成立', '于', '1836年', '3月', '1日', '，', '县名', '纪念', '第六', '任', '总 统',
    #      '约翰·昆西·亚当斯'], ['第一', '楠', '主角', '是', '哪一天', '发行', '的']))
    # print(bleu1("", "1957年"))
    # print(list(jieba.cut("高照容13岁入宫，文明太后见其貌美，遂将她送给孝文帝。")))
    # answer = extract_answer(
    #     ['相应', '地', '，', '行文', '关系', '也', '可', '分为', '上', '行文', '关系', '、', '平行文', '关系', '和', '下', '行文', '关系', '三', '种',
    #      '。'],
    #     ['多', '级', '行文', '中', '行文', '关系', '可以', '分为', '哪', '三', '种'])
    # print(answer)
