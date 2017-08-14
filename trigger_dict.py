import pandas as pd
import numpy as np
import os
import util
from collections import Counter


class TriggerModel:
    def __init__(self, word, pos, relations):
        self.word = word
        self.pos = pos
        self.relations = relations

    def __eq__(self, other):
        return self.word == other.word and self.pos == other.pos and self.relations == other.relations

    def __hash__(self):
        return hash(self.word)


class TriggerDict:
    def __init__(self, path=None):
        if path is not None:
            self.triggers = self.build_triggers(path)
        else:
            self.triggers = None

    def save_trigger_model_as_csv(self, save_name):
        trigger = [(word, model.pos, model.relations, self.triggers[word][model]) for word in self.triggers for model in self.triggers[word]]
        trigger_df = pd.DataFrame(np.array(trigger), columns=['触发词', '词性', '依存关系', '出现频数'])
        trigger_df.to_csv(save_name, encoding='utf-8')

    def show_info(self):
        x = self.triggers
        total_num = sum([sum(x[i].values()) for i in x])
        total_model = sum([len(x[i]) for i in x])
        more_than_one_word_num = len([i for i in x if sum(x[i].values()) > 1])
        more_than_one_model_num = len([a for i in x for a in x[i].items() if a[1] > 1])
        trigger_word_num = len(x)
        print('一共{0}个句子中，抽取出{1}个触发词，其中出现不止一次的有{2}个词，占{3}。抽取出来{4}个触发词模板，出现不止一次的模板的词有{5}，占{6}'.
              format(total_num, trigger_word_num, more_than_one_word_num, more_than_one_word_num / trigger_word_num,
                     total_model, more_than_one_model_num, more_than_one_model_num / total_model))

    def trigger_model_check(self, target_parse_model, level=1):
        target_word = target_parse_model['word']
        if target_word not in self.triggers:
            return 0
        else:
            model_counter = self.triggers[target_word]
            for model in model_counter:
                if model == target_parse_model:
                    return model_counter[model] + 10
                else:
                    return 1

    @staticmethod
    def build_triggers(path):
        """ 利用语料句构建触发词库，path为语料文件路径，语料文件每一行存储一句。 """
        util.load_model()
        trigger_dict = {}
        with open(path, 'r', encoding='utf-8') as f:
            sentences = f.read().splitlines()
        for s in sentences:
            try:
                td = util.parse_tree_by_sentence(s)
                word = td['word']
                trigger_model = TriggerModel(td['word'], td['pos'], td['relations'])
                if word not in trigger_dict:
                    trigger_dict[word] = [trigger_model]
                else:
                    trigger_dict[word].append(trigger_model)
            except Exception as e:
                print(e)
                continue
        util.release_model()
        for word in trigger_dict:
            trigger_dict[word] = Counter(trigger_dict[word])
        return trigger_dict

if __name__ == '__main__':
    pass

