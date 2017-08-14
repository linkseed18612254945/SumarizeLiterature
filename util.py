import pandas as pd
import literature
import pyltp
import pickle
import os
from trigger_dict import TriggerDict
from math import inf

MIN_SENTENCE_NUM = 140
STOP_WORD_PATH = './相关词表/停用词词表.txt'
LTP_SEGMENT_MODE = './LTP_model/cws.model'
LTP_POS_MODE = './LTP_model/pos.model'
LTP_PARSE_MODE = './LTP_model/parser.model'
SEGMENTOR = pyltp.Segmentor()
POSTARGGER = pyltp.Postagger()
PARSER = pyltp.Parser()
with open('./相关词表/线索词词表.txt', 'r', encoding='utf-8') as f:
    CLUE_WORDS = f.read().splitlines()


def load_model():
    """ 加载LTP包的分词、词性标注、句法分析模型 """
    SEGMENTOR.load(LTP_SEGMENT_MODE)
    POSTARGGER.load(LTP_POS_MODE)
    PARSER.load(LTP_PARSE_MODE)


def release_model():
    """ 释放LTP包的分词、词性标注、句法分析模型 """
    SEGMENTOR.release()
    POSTARGGER.release()
    PARSER.release()


def cut_sent(words, punt_list):
    """ 自定义短句切分函数 """
    start = 0
    i = 0
    sents = []
    for word in words:
        if word in punt_list:
            sents.append(words[start:i])
            start = i + 1
            i += 1
        else:
            i += 1
    if start < len(words):
        sents.append(words[start:])
    return sents


class AttrDict(dict):
    """ 自定义字典结构 """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def list_search(l, n):
    """ 列表查询方法，如果L中存在n则返回n所在位置序号，否则返回正无穷值 """
    if n not in l:
        return inf
    else:
        return l.index(n)


def min_from_dict(d, x):
    """ """
    category = min([(list_search(d[k], x), k) for k in d])
    if category[0] != inf:
        return category[1]
    else:
        return '未分类'


def clue_score(words):
    for word in words:
        if word in CLUE_WORDS:
            return 1
    return 0


def pickle_object(name, obj):
    """ 对象打包存储 """
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(path):
    """ 打包对象读取 """
    with open(path, 'rb') as f:
        os = pickle.load(f)
    return os


def filter_sentence(sentences, min_len):
    """ 只保留句子长度大于指定值的句子 """
    return [s for s in sentences if len(s) > min_len]


def remove_stop_words(words, path):
    """ 针对所有任务都进行的去除停用词和数字 """
    with open(path, 'r', encoding='utf-8') as f:
        stop_words = set(f.read().splitlines())
    return [word for word in words if word not in stop_words and not word.isdigit()]

def tokenize(sentence):
    words = SEGMENTOR.segment(sentence)
    return list(words)


def parse_tree_by_tuple(words, pos, parse):
    head = -1
    for i, word in enumerate(words):
        if parse[i][0] == 'HED':
            head = i + 1
    relations = []
    use_relations = []
    for i, word in enumerate(words):
        if parse[i][1] == head:
            relations.append((i + 1, word, pos[i], parse[i][0]))
            use_relations.append((parse[i][0], pos[i]))
    trigger_pattern = AttrDict(word=words[head - 1], pos=pos[head - 1], relations=set(use_relations))
    return trigger_pattern


def parse_tree_by_sentence(sentence):
    """ 将句子解析为句法树并返回触发词模板 """
    words = list(SEGMENTOR.segment(sentence))
    pos = list(POSTARGGER.postag(words))
    parse_result = list(PARSER.parse(words, pos))
    print([(word, parse_result[i].head, parse_result[i].relation) for i, word in enumerate(words)])
    head = -1
    for i, word in enumerate(words):
        if parse_result[i].relation == 'HED':
            head = i + 1
    relations = []
    use_relations = []
    for i, word in enumerate(words):
        if parse_result[i].head == head:
            relations.append((i + 1, word, pos[i], parse_result[i].relation))
            use_relations.append((parse_result[i].relation, pos[i]))
    trigger_pattern = AttrDict(word=words[head - 1], pos=pos[head - 1], relations=set(use_relations))
    return trigger_pattern


def data_pre_process(path, back_up_path, pickle_data=True):
    load_model()
    with open(path, 'r', encoding='utf-8') as f:
        files = f.read().splitlines()
    literatures = []
    for file in files:
        title = file.split('\t')[0]
        content = file.split('\t')[1]
        sentences = cut_sent(content, punt_list='。！？;；：')
        content_sentence = []
        content_words = []
        content_pos = []
        content_parse = []
        for sentence in sentences:
            if sentence == '':
                continue
            shorts = cut_sent(sentence, punt_list=',，')
            shorts = [short for short in shorts if short != '']
            if shorts == []:
                continue
            content_sentence.append(shorts)
            shorts_words = [list(SEGMENTOR.segment(s)) for s in shorts]
            content_words.append(shorts_words)
            shorts_pos = [list(POSTARGGER.postag(word)) for word in shorts_words]
            content_pos.append(shorts_pos)
            shorts_parse_temp = [PARSER.parse(shorts_words[i], shorts_pos[i]) for i in range(len(shorts_words))]
            shorts_parse = [[(p.relation, p.head) for p in sentence_parse] for sentence_parse in shorts_parse_temp]
            content_parse.append(shorts_parse)
        literatures.append(literature.LiteratureFile(title, content_sentence, content_words, content_pos, content_parse))
    print('预处理完成')
    if pickle_data:
        pickle_object(back_up_path, literatures)
        print('已将处理好的文献打包存储')
    release_model()


def pickle_trigger_dict(trigger_basic_path, pickle_path):
    trigger = dict()
    trigger['结果'] = TriggerDict(os.path.join(trigger_basic_path, '结果.txt'))
    trigger['方法'] = TriggerDict(os.path.join(trigger_basic_path, '方法.txt'))
    trigger['目的'] = TriggerDict(os.path.join(trigger_basic_path, '目的.txt'))
    pickle_object(pickle_path, trigger)


def clean_stop_words(words):
    with open(STOP_WORD_PATH, 'r', encoding='utf-8') as f:
        stop_words = f.read().splitlines()
    words = [word for word in words if word not in stop_words]
    return words

def process_example(sentence):
    words = list(SEGMENTOR.segment(sentence))
    pos = list(POSTARGGER.postag(words))
    return [(words[i], pos[i]) for i in range(len(words))]


def data_pre_process_from_csv(path, back_up_path, pickle_data=True):
    """ 读取文献数据，加载LTP处理模块，对文本进行预处理， 返回一个Literature类的集合，同时以pickle方式存储备份"""
    load_model()
    df = pd.read_csv(path, encoding='utf-8')
    titles = df['Title'].values
    contents = df['content_all'].values
    literatures = []
    for i, title in enumerate(titles):
        content = contents[i]
        sentences = cut_sent(content, punt_list='。！？;；：')
        if len(sentences) < 80:
            continue
        content_sentence = []
        content_words = []
        content_pos = []
        content_parse = []
        for sentence in sentences:
            shorts = cut_sent(sentence, punt_list=',，')
            if not shorts:
                continue
            content_sentence.append(shorts)
            shorts_words = [list(SEGMENTOR.segment(s)) for s in shorts]
            content_words.append(shorts_words)
            shorts_pos = [list(POSTARGGER.postag(word)) for word in shorts_words]
            content_pos.append(shorts_pos)
            shorts_parse_temp = [PARSER.parse(shorts_words[i], shorts_pos[i]) for i in range(len(shorts_words))]
            shorts_parse = [[(p.relation, p.head) for p in sentence_parse] for sentence_parse in shorts_parse_temp]
            content_parse.append(shorts_parse)
        literatures.append(literature.LiteratureFile(title, content_sentence, content_words, content_pos, content_parse))

    if pickle_data:
        pickle_object(back_up_path, literatures)
        print('已将处理好的文献打包存储')


