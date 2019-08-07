#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: sunyueqing
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: sunyueqinghit@163.com
@File : test.py
@Time : 2019/5/24 21:56
@Site : 
@Software: PyCharm
'''
import os
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer
import data_path

LTP_DATA_DIR = 'D:/ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
segmentor = Segmentor()  # 初始化实例
segmentor.load_with_lexicon(cws_model_path, data_path.user_dict)
postagger = Postagger()
postagger.load(os.path.join(LTP_DATA_DIR, "pos.model"))
parser = Parser()
parser.load(os.path.join(LTP_DATA_DIR, "parser.model"))
recognizer = NamedEntityRecognizer()
recognizer.load(os.path.join(LTP_DATA_DIR, "ner.model"))  # 加载命名实体识别模型
sent = "《天上人间》这个游戏总共开发了多久"
print(list(segmentor.segment(sent)))
# 命名实体
words = segmentor.segment(sent)
# print(list(words))
postags = postagger.postag(words)
netag = recognizer.recognize(words, postags)
for word, ntag in zip(words, netag):
    if ntag != 'O':
        print(word + '/' + ntag)
print("\t".join(netag))

# print(''.join(["吸入可能由于喉、支气管的痉挛、水肿、炎症、化学性肺炎、肺水肿而致死。", "中毒表现有烧灼感、咳嗽、喘息、喉炎、气短、头痛、恶心、呕吐。", "中毒表现有烧灼感、咳嗽、喘息、喉炎、气短、头痛、恶心、呕吐。"]))
