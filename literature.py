import util
import text_rank
import pandas as pd


class LiteratureFile:
    """
    文献文档类，每个LiteratureFile类代表一篇文献, 类属性包括:
    title: 文献标题
    keyword: 文献自带关键字
    abstract: 文献自带摘要文本
    content_sentences: 文献正文句子，一维List存储
    content_words: 文献正文对应的词序列，二维List存储，每个List对应一个句子
    content_pos: 词序列对应的词性序列，二维List存储，与词序列一一对应
    content_parse_relation: 词序列对应的词在句中成分, 二维List存储，与词序列一一对应
    content_parse_head:  词序列对应的词在句中的关联词序号，如1表示第一个词，其中0为HEAD词, 二维List存储，与词序列一一对应
    
    """
    def __init__(self, title, content, words, pos, parse):
        self.title = title
        self.content_sentences = content
        self.article_length = len(content)
        self.content_words = words
        self.content_pos = pos
        self.content_parse = parse
        self.trigger_dict = {}

    @staticmethod
    def combine_words(content):
        """ 将短句词序列组转换为长句词序列 """
        new_content = []
        for sentence in content:
            s = []
            for short in sentence:
                s += short
            new_content.append(s)
        return new_content

    def sorted_sentences(self):
        """ 返回按关键度排序的句子字典序列 """
        return text_rank.sort_sentences([','.join(i) for i in self.content_sentences], self.combine_words(self.content_words))

    def set_trigger_dict(self, trigger_dict):
        self.trigger_dict = trigger_dict

    def sentence_total_score(self, sentence_score):
        total_score = {}
        for short_score in sentence_score:
            for tp in short_score:
                if tp not in total_score:
                    total_score[tp] = short_score[tp]
                else:
                    total_score[tp] += short_score[tp]
        return total_score

    def sentences_scores(self):
        sentence_scores = []
        for i, sentence in enumerate(self.content_sentences):
            sentence_score = []
            for j, short in enumerate(sentence):
                short_score = {}
                short_parse_tree = util.parse_tree_by_tuple(self.content_words[i][j], self.content_pos[i][j],
                                                            self.content_parse[i][j])
                for tp in self.trigger_dict:
                    short_score[tp] = self.trigger_dict[tp].trigger_model_check(short_parse_tree)
                sentence_score.append(short_score)
            sentence_scores.append(self.sentence_total_score(sentence_score))
        return sentence_scores

    def sentences_classify(self, sored_scores):
        category_sentence = {'方法': [], '目的': [], '结果': [], '其他': []}
        for i, score in enumerate(self.sentences_scores()):
            sentence = ','.join(self.content_sentences[i])
            words = []
            for short_words in self.content_words[i]:
                words += short_words
            category = max(score, key=lambda x: score[x])
            max_score = score[category]
            item = util.AttrDict(sentence=sentence, sorted_score=sored_scores[i]['weight'],
                                 words=words, category_score=max_score, category=category)
            if max_score >= 1:
                category_sentence[category].append(item)
            else:
                category_sentence['其他'].append(item)
        for category in category_sentence:
            category_sentence[category].sort(key=lambda item: item['sorted_score'], reverse=True)
        return category_sentence

    def create_auto_abstract(self, s_num, output_path):
        result = self.sentences_classify(self.sorted_sentences())
        sentences = []
        for k in result:
            sentences += k
            sentences += [item['sentence'] + '。' for item in result[k]][:s_num]
        abstract = '\n'.join(sentences)
        print('共生成{0}句的摘要'.format(len(sentences)))
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(abstract)

    def create_rouge_result(self, s_num, output_path):
        result = self.sentences_classify(self.sorted_sentences())
        words = []
        for k in result:
            for item in result[k][:s_num]:
                words.append(' '.join(util.clean_stop_words(item['words'])))
        abstract = '\n'.join(words)
        print('共生成{0}句的摘要'.format(len(words)))
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(abstract)

    def create_baseline(self, s_num, output_path):
        ss = self.sorted_sentences()
        words = [' '.join(ss[i]['words']) for i in range(s_num)]
        abstract = '\n'.join(words)
        print('共生成{0}句的摘要'.format(len(words)))
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(abstract)

    def create_result(self):
        result = self.sentences_classify(self.sorted_sentences())

    def save_scores_as_csv(self, save_name, num):
        sorted_scores = self.sorted_sentences()
        result = self.sentences_classify(sorted_scores)
        result_matrix = []
        for cate in result:
            result_matrix += result[cate][:num]
        result_df = pd.DataFrame(result_matrix)
        result_df.to_csv(save_name, encoding='utf-8')

