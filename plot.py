import matplotlib.pyplot as plt
import numpy as np


def draw_bar(name, recalls):
    labels = [i * 2 for i in range(1, len(recalls) + 1)]
    plt.bar(range(len(recalls)), recalls, fc='gray', tick_label=labels)
    plt.xlabel('Article ID')
    plt.ylabel('Rouge Score')
    plt.title('{0}'.format(name.upper()))
    plt.savefig('./结果图片/结果{0}.png'.format(name.upper()))
    plt.show()


def compete_bar(baseline, hyp):
    size = 3
    x = np.arange(size)
    a = [hyp['rouge-1'], hyp['rouge-2'], hyp['rouge-l']]
    b = [baseline['rouge-1'], baseline['rouge-2'], baseline['rouge-l']]
    total_width, n = 0.8, 2
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.bar(x, b, width=width, label='Baseline', fc='blue')
    plt.bar(x + width, a, width=width, label='EventBased', fc='red', tick_label=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
    plt.xlabel('RougeType')
    plt.ylabel('Score')
    plt.title('Baseline Compare')
    plt.legend()
    plt.savefig('./结果图片/比较.png')
    plt.show()


def draw_string(rouges):
    y_rouge1 = rouges['rouge-1']
    y_rougel = rouges['rouge-l']
    y_rouge2 = rouges['rouge-2']
    x = list(range(1, len(y_rouge1) + 1))
    plt.plot(x, y_rouge1, '', label='ROUGE-1')
    plt.plot(x, y_rouge2, '', label='ROUGE-2')
    plt.plot(x, y_rougel, '', label='ROUGE-l')
    plt.title('Rouges')
    plt.legend(loc='upper right')
    plt.xlabel('月份')
    plt.ylabel('XX事件数')
    # plt.savefig('./结果图片/变化.png')
    plt.show()

