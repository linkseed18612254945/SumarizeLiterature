import util
import literature
from trigger_dict import TriggerDict
import text_rank
import os
import rouge_evaluate

DATA_PATH = './语料/现代图书情报技术.txt'
DATA_BACKUP_PATH = './语料/预处理后正文语料'
TRIGGER_PATH = './语料/触发词语料'
TRIGGER_SAVE_PATH = './触发词抽取结果'
TRIGGER_BACKUP_PATH = './语料/处理后触发词库'
OUTPUT_PATH = './ResultForROUGE/hyp'
BASELINE_PATH = './ResultForROUGE/baseline'
RESUlT_PATH = './test.csv'
ABSTRACT_RESULT_PATH = './自动摘要结果'
RESUlT_BACK_UP_PATH = './语料/rouge结果'


def show_result(test):
    r = test.sorted_sentences()
    t = test.sentences_classify(r)
    for i in t:
        print('-----------------', i, '---------------------')
        y = sorted(t[i], key=lambda item: item['category_score'], reverse=True)
        for x in y:
            print(x)


def get_compares():
    avg_rouges = {'rouge-1': [], 'rouge-l': [], 'rouge-2': []}
    for n in range(1, 6):
        for i, a in enumerate(ls):
            a.create_rouge_result(n, os.path.join(OUTPUT_PATH, 'hyp_{0}.txt'.format(i + 1)))
        avg_rouge = rouge_evaluate.get_avg_recall_rouge()
        for k in avg_rouges:
            avg_rouges[k].append(avg_rouge[k])
    print(avg_rouges)
    util.pickle_object(RESUlT_BACK_UP_PATH, avg_rouges)

if __name__ == '__main__':
    # 自动处理正文语料并打包
    # util.data_pre_process(DATA_PATH, DATA_BACKUP_PATH, pickle_data=True)

    # 获取单个类比的触发词库保存为CSV并给出统计信息(用于结果展示)
    # tr = TriggerDict(os.path.join(TRIGGER_PATH, '结果.txt'))
    # tr.save_trigger_model_as_csv(os.path.join(TRIGGER_SAVE_PATH, '结果.csv'))
    # tr.show_info()

    # 直接打包三类触发词库结果
    # util.pickle_trigger_dict(TRIGGER_PATH, TRIGGER_BACKUP_PATH)

    # 读取已打包的触发词库，用字典信息存储
    triggers = util.pickle_load(TRIGGER_BACKUP_PATH)

    # 读取打包的正文语料
    ls = util.pickle_load(DATA_BACKUP_PATH)
    for article in ls:
        article.set_trigger_dict(triggers)

    # 创建用于测评的自动摘要结果
    for i, a in enumerate(ls):
        # a.create_baseline(12, os.path.join(BASELINE_PATH, 'baseline_{0}.txt'.format(i + 1)))
        a.create_auto_abstract(5, os.path.join(ABSTRACT_RESULT_PATH, '{0}.txt'.format(a.title)))
        # a.create_rouge_result(3, os.path.join(OUTPUT_PATH, 'hyp_{0}.txt'.format(i + 1)))
    # recalls_rouge = [rouge_result[i] for i in range(0, len(rouge_result), 2)]






