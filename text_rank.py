import os
import util
import math
import numpy as np
import networkx as nx


def default_sentence_similarity(sentence1, sentence2):
    """ 默认用于计算两个句子之间的相似度， 每个句子由一个词序列组成 """
    words = list(set(sentence1 + sentence2))
    vector1 = np.array([float(sentence1.count(word)) for word in words])
    vector2 = np.array([float(sentence2.count(word)) for word in words])
    vector3 = vector1 * vector2
    vector4 = [1 for num in vector3 if num > 0]

    co_occur_num = sum(vector4)
    if abs(co_occur_num) <= 1e-12:
        return 0

    denominator = math.log(float(len(sentence1))) + math.log(float(len(sentence2)))
    if abs(denominator) <= 1e-12:
        return 0
    return co_occur_num / denominator


def similarity_matrix(source, sim_func):
    """ 计算相似度矩阵 """
    source_num = len(source)
    graph = np.zeros((source_num, source_num))
    for x in range(source_num):
        for y in range(x, source_num):
            graph[x][y] = sim_func(source[x], source[y])
            graph[y][x] = sim_func(source[x], source[y])
    return graph


def sort_sentences(sentences, words, sim_func=default_sentence_similarity):
    """ 将句子按关键程度进行排序 """
    sorted_sentences = {}
    graph = similarity_matrix(words, sim_func)
    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph)
    average_score = sum(scores.values()) / len(scores)
    for index, score in scores.items():
        feature_score = util.clue_score(words[index]) * average_score + score
        if len(words[index]) < 8:
            feature_score = 0
        item = util.AttrDict(sentence=sentences[index], weight=feature_score, words=util.clean_stop_words(words[index]))
        sorted_sentences[index] = item
    return sorted_sentences
