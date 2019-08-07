#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: sunyueqing
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: sunyueqinghit@163.com
@File : preprocessed.py
@Time : 2019/5/9 16:37
@Site : 
@Software: PyCharm
'''

import os
from pyltp import Segmentor
import json
import time
import data_path
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.analysis import KeywordAnalyzer
from whoosh.qparser import syntax
# from jieba.analyse import ChineseAnalyzer
import jieba
from whoosh.analysis import Tokenizer, Token

LTP_DATA_DIR = 'D:/ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`


class ChineseTokenizer(Tokenizer):
    def __call__(self, value, positions=False, chars=False,
                 keeporiginal=False, removestops=True,
                 start_pos=0, start_char=0, mode='', **kwargs):
        # assert isinstance(value, text_type), "%r is not unicode" % value
        t = Token(positions, chars, removestops=removestops, mode=mode,
                  **kwargs)
        seglist = jieba.cut(value, cut_all=True)
        for w in seglist:
            t.original = t.text = w
            t.boost = 1.0
            if positions:
                t.pos = start_pos + value.find(w)
            if chars:
                t.startchar = start_char + value.find(w)
                t.endchar = start_char + value.find(w) + len(w)
            yield t


def ChineseAnalyzer():
    return ChineseTokenizer()


def segment():
    '''
    对比使用LTP和jieba的效果
    :return:
    '''
    # segmentor = Segmentor()  # 初始化实例
    # segmentor.load(cws_model_path)  # 加载模型
    # 读取未分词文件
    with open(data_path.passages_multi_sentences, encoding='utf-8') as fin:
        read_results = [json.loads(line.strip()) for line in fin.readlines()]
    for result in read_results:
        # result['document'] = [' '.join(segmentor.segment(sent)) for sent in result['document']]
        result['document'] = [' '.join(jieba.cut(sent)) for sent in result['document']]
    # segmentor.release()  # 释放模型
    print("分词结束，开始写入文件...")
    # 写回分词后的文件
    with open(data_path.passages_segment, 'w', encoding='utf-8') as fout:
        for item in read_results:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')


def create_index():
    # 定义schema
    schema = Schema(id=ID(stored=True), document=TEXT(stored=False))  # , analyzer=ChineseAnalyzer()
    ix = create_in(data_path.index_dir, schema)
    writer = ix.writer()
    # 读取json文件
    with open(data_path.passages_segment, 'r', encoding='utf-8') as fin:
        docs = [json.loads(line.strip()) for line in fin.readlines()]
    # 加入索引
    start = time.time()
    for doc in docs:
        writer.add_document(id=str(doc['pid']), document=' '.join(doc['document']))
    writer.commit()
    end = time.time()
    print("成功建立索引，用时{}s".format(end - start))  # 345s


def search():
    index = open_dir(data_path.index_dir)
    with index.searcher() as searcher:
        parser = QueryParser("document", schema=index.schema, group=syntax.OrGroup)
        query = parser.parse("罗静恩 韩文 名字 是 什么")
        results = searcher.search(query)
        for hit in results:
            print(hit)  # hit.highlights("document")没有store


def train_test():
    # segmentor = Segmentor()  # 初始化实例
    # segmentor.load(cws_model_path)  # 加载模型
    # 读取训练文件
    with open(data_path.train, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    # 打开索引文件
    index = open_dir(data_path.index_dir)
    start = time.time()
    time1 = time.time()
    with index.searcher() as searcher:
        parser = QueryParser("document", index.schema, group=syntax.OrGroup)
        pid_label = []
        pid_pre = []
        i = 0
        for item in items:
            pid_label.append(item['pid'])  # 训练文件中的pid
            # q = ' '.join(segmentor.segment(item['question']))  # question进行分词
            q = ' '.join(jieba.cut(item['question']))
            results = searcher.search(parser.parse(q))
            if len(results) > 0:
                pid_pre.append([int(res['id']) for res in results[0:3]])  # Top3
                # pid_pre.append([int(results[0]['id'])])  # Top1
            else:
                pid_pre.append([])
            i += 1
            if i % 100 == 0:
                data_path.logging.info("查询 {} 完成, 用时 {}s".format(i, time.time() - time1))
                print("查询 {} 完成, 用时 {}s".format(i, time.time() - time1))
                time1 = time.time()
    end = time.time()
    print("查询 {}, 用时 {}s".format(i, end - start))
    data_path.logging.info("查询 {}, 用时 {}s".format(i, end - start))
    eval(pid_label, pid_pre)


def eval(label, pre):
    # print(label, "/", pre)
    rr = 0  # 检索回来的相关文档数
    rr_rn = len(label)  # 检索回来的文档总数
    for i in range(len(label)):
        if label[i] in pre[i]:
            rr += 1
        else:
            print(label[i], ":", pre[i])
    p = float(rr) / rr_rn

    print(
        "总计:{}, 检索回来的相关文档数:{}, 检索回来的文档总数:{}, Precision:{}".format(len(label), rr, rr_rn, p))
    data_path.logging.debug(
        "总计:{}, 检索回来的相关文档数:{}, 检索回来的文档总数:{}, Precision:{}".format(len(label), rr, rr_rn, p))


if __name__ == '__main__':
    # segment()
    # create_index()
    # search()
    train_test()
