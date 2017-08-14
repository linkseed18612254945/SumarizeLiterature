import os
import util


def prep(path):
    fnames = os.listdir(path)
    texts = []
    for title in fnames:
        try:
            with open(os.path.join(path, title), 'r', encoding='utf-8') as f:
                text = f.read()
            text = text.replace('\n', '')
            text = text.replace(' ', '')
            text = text.replace('\t', '')
            texts.append(title[:title.find('.txt')] + '\t' + text)
            print(len(text), title[:title.find('.txt')], text)
        except UnicodeDecodeError:
            print(title)
    with open('现代图书情报技术.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(texts))


if __name__ == '__main__':
    base_path = r'C:\Users\51694\PycharmProjects\SumarizeByEvent\现代图书情报技术正文'
    prep(base_path)


