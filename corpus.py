import pyltp
import re
import random
import logging
import numpy as np
from collections import OrderedDict
import pickle
import os
from collections import Counter


class FileItem:
    """ 文档类，每个文档被存储为一个文档类，默认保存有词序列、文档类别、文档id、词性序列四个属性。可以根据需求添加新属性"""

    def __init__(self, id, category, words, pos, parse_result):
        self.id = id
        self.category = category
        self.words = words
        self.pos = pos
        self.parse_result = parse_result

    def __repr__(self):
        return '{0} {1}\n{2}\n{3}\n'.format(self.id, self.category, self.words, self.pos)


class WordItem:
    """ 词属性类，保存有一个词出现的文档号集合以及一个词在某类里出现的总频次 """

    def __init__(self):
        self.file_ids = set()
        self.count = 0
        self.chi = 0

    def add_word(self, file_id):
        self.file_ids.add(file_id)
        self.count += 1

    def __repr__(self):
        return str(self.count) + ' ' + str(self.file_ids)


class BagOfWords:
    """ 词袋模型类, 每个语料库对应生成一个词袋，词袋类负责支持特征筛选和对词进行编号，对文档类进行向量化 """

    def __init__(self, words_dict=None, file_count=None):
        self.dict = words_dict
        self.file_count = file_count

    def bow_features(self, feature_model, frequency):
        """ 
            特征选择，根据需求选择特定词典作为文档特征, 参数feature_model用于设定特征选择方法
            Total: 将全部词作为训练特征
            Frequency: 选择每个类别中出现频率前 n% 的词作为训练特征，用frequency参数进行设定，默认为50%
        """
        words = set()
        if feature_model == 'Total':
            words = self.__total_words()
        elif feature_model == 'Frequency':
            words = self.__frequency_words(frequency)
        else:
            pass
        return self.dict_with_id(words)

    def __total_words(self):
        chosen_words = []
        for category in self.dict:
            chosen_words += list(self.dict[category].keys())
        return set(chosen_words)

    def __frequency_words(self, frequency):
        chosen_words = []
        for category in self.dict:
            category_num = int(len(self.dict[category].keys()) * frequency)
            chosen_items = sorted(self.dict[category].items(), key=lambda w: w[1].count, reverse=True)[:category_num]
            chosen_words += [item[0] for item in chosen_items]
        return set(chosen_words)

    def __chi_words(self, frequency):
        chosen_words = []
        for category in self.dict:
            category_num = int(len(self.dict[category].keys()) * frequency)
            chosen_items = sorted(self.dict[category].items(), key=lambda w: w[1].count, reverse=True)[:category_num]
            chosen_words += [item[0] for item in chosen_items]

    @staticmethod
    def dict_with_id(words):
        """ 对词进行编号，使一个维度对应一个词 """
        id_dict = OrderedDict()
        for idx, word in enumerate(words):
            id_dict[word] = idx
        return id_dict

    def save_bow(self, bow_name='saved_BoW'):
        """ 存储词袋模型对象在一个新建目录下，包含两个字典数据 """
        os.mkdir(bow_name)
        with open('%s/words_dict' % bow_name, 'wb') as f:
            pickle.dump(self.dict, f)
        with open('%s/file_count' % bow_name, 'wb') as f:
            pickle.dump(self.file_count, f)

    def load_bow(self, bow_name='saved_BoW'):
        """ 读取已存储的词袋模型字典，要求读取目录包含words_dict, file_count两个Pickle字典 """
        try:
            with open('%s/words_dict' % bow_name, 'rb') as f:
                self.dict = pickle.load(f)
            with open('%s/file_count' % bow_name, 'rb') as f:
                self.file_count = pickle.load(f)
        except IOError as e:
            raise e


class Corpus:
    """ 
    语料库对象负责将原始文档读入内存，完成分词和词性标注操作，同时创建对应的词袋模型
    原始语料格式，每个待分类文档保存为一个独立的文件(默认为txt格式，可指定读取特定后缀)。一类文档放在同一个文件夹中，文件夹名即为类别名称。
    通过file_vectors()方法得到向量化后的文档，然后可以进行后续的分类训练
    """

    def __init__(self, path, file_end='txt'):
        self.category_ids = {}
        self.dir_path = path
        self.file_end = '.' + file_end
        self.file_total_num = len(self.file_paths())
        self.files = self.build_files()

    def file_paths(self):
        """ 遍历指定目录，获取文档路径 """
        return [os.path.join(dirname, file) for (dirname, dirs, files) in os.walk(self.dir_path) for file in files
                if file.lower().endswith(self.file_end)]

    def build_files(self):
        """ 遍历原始文档，进行分词词性标注，去除停用词等，创建FileItem类集合 """
        files = []
        category_id = 0
        segmentor = pyltp.Segmentor()
        segmentor.load(r'C:\Users\51694\PycharmProjects\paper\ltp_model\cws.hyp')
        postagger = pyltp.Postagger()
        postagger.load(r'C:\Users\51694\PycharmProjects\paper\ltp_model\pos.hyp')
        parser = pyltp.Parser()
        parser.load(r'C:\Users\51694\PycharmProjects\paper\ltp_model\parser.hyp')
        for ids, path in enumerate(self.file_paths()):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    category = self.path2category(path)
                    if category not in self.category_ids:
                        self.category_ids[category] = category_id
                        category_id += 1
                    raw = self.process_line(f.read())
                    words = self.remove_stop_words(list(segmentor.segment(raw)))
                    words = self.clean_specific(words)
                    pos = list(postagger.postag(words))
                    parse_result = list(parser.parse(words, pos))
                    files.append(FileItem(ids, category, words, pos, parse_result))
                except UnicodeDecodeError:
                    logging.warning(path + ' UTF-8解码失败，请检查文本格式')
                    continue
        segmentor.release()
        postagger.release()
        parser.release()
        return files

    def build_bow(self):
        """ 利用已创建的FileItem对象集合创建语料库对应的词袋对象 """
        bow_dict = {}
        bow_file_count = Counter()
        for file in self.files:
            bow_file_count[file.cate] += 1
            if file.cate not in bow_dict:
                bow_dict[file.cate] = {}
            for word in file.words:
                if word not in bow_dict[file.cate]:
                    bow_dict[file.cate][word] = WordItem()
                bow_dict[file.cate][word].add_word(file.id)
        return BagOfWords(bow_dict, bow_file_count)

    def files_data(self, bow, file_num, feature_model='Total', frequency=0.5):
        """ 获取向量化后的文档和对应类别标签数据，可以利用file_num参数指定文档数量，feature_mode指定向量化方法。该方法是提供训练使用的API。 """
        random.shuffle(self.files)
        files = self.files[:file_num]
        bow_dict = bow.bow_features(feature_model, frequency)
        file_vectors = []
        file_labels = []
        for file in files:
            file_vectors.append(self.file_to_vector(file, bow_dict))
            file_labels.append(self.category_ids[file.cate])
        return file_vectors, file_labels

    @staticmethod
    def file_to_vector(file, bow_dict):
        """ 将文档对象向量化 """
        file_vector = np.zeros(len(bow_dict))
        for word in file.words:
            if word in bow_dict:
                file_vector[bow_dict[word]] += 1
        return file_vector

    @staticmethod
    def process_line(line):
        """ 去除原始文本中的特殊符号，提高分词准确度 """
        return re.sub("]-·[\s+\.\!\/_,$%^*(+\"\':]+|[+——！，。？、~@#￥%……&*（）():\"=《\\n]+", " ", line)

    @staticmethod
    def remove_stop_words(words):
        """ 针对所有任务都进行的去除停用词和数字 """
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            stop_words = set(f.read().splitlines())
        return [word for word in words if word not in stop_words and not word.isdigit()]

    @staticmethod
    def clean_specific(words):
        """ 针对特定任务对词进行筛选 """
        def is_english(w):
            return all([ord(c) < 128 for c in w])
        words_copy = words.copy()
        for word in words:
            if is_english(word) or len(word) < 2 or len(word) > 6:
                words_copy.remove(word)
        return words_copy

    @staticmethod
    def path2category(path):
        """ 根据文档路径获得文档所属类别 """
        reverse_path = path[::-1]
        if '\\' in path:
            symb = '\\'
        else:
            symb = '/'
        temp = reverse_path[reverse_path.find(symb) + 1:]
        return temp[:temp.find('\\')][::-1]
