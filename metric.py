#!/usr/bin/python
# coding:utf-8

"""
@author: Mingxiang Tuo
@contact: tuomx@qq.com
@file: metric.py
@time: 2019/6/1 15:59
实验3.4 评价方式包含三个指标，主要看字符级别的bleu1值，其他供参考
1. precision，recall，F1值：取所有开发集上的平均
2. EM（exact match）值：精确匹配的准确率
3. 字符级别的bleu1值
"""

from collections import Counter
from nltk.translate.bleu_score import sentence_bleu


def precision_recall_f1(prediction, ground_truth):
    """
    计算预测答案prediction和真实答案ground_truth之间的字符级别的precision，recall，F1值，
    Args:
        prediction: 预测答案（未分词的字符串）
        ground_truth: 真实答案（未分词的字符串）
    Returns:
        floats of (p, r, f1)
    eg:
    >>> prediction = '北京天安门'
    >>> ground_truth = '天安门'
    >>> precision_recall_f1(prediction, ground_truth)
    >>> (0.6, 1.0, 0.7499999999999999)
    """
    #     # 对于中文字符串，需要在每个字之间加空格
    prediction = " ".join(prediction)
    ground_truth = " ".join(ground_truth)

    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def exact_match(all_prediction, all_ground_truth):
    """
    计算所有预测答案和所有真实答案之间的准确率
    Args:
        all_prediction: 所有预测答案（数组）
        all_ground_truth: 所有真实答案（数组）
    Returns:
        floats of em
    eg:
    >>> all_prediction = ['答案A', '答案B', '答案C']
    >>> all_ground_truth = ['答案A', '答案B', '答案D']
    >>> exact_match(all_prediction, all_ground_truth)
    >>> 0.6666666666666666
    """
    assert len(all_prediction) == len(all_ground_truth)
    right_count = 0
    for pred_answer, true_answer in zip(all_prediction, all_ground_truth):
        if pred_answer == true_answer:
            right_count += 1
    return 1.0 * right_count / len(all_ground_truth)


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
