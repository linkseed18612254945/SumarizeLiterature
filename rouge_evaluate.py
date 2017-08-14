from rouge import FilesRouge
import plot
import os

hyp_path = './ResultForROUGE/hyp'
baseline_path = './ResultForROUGE/baseline'
ref_path = './ResultForROUGE/ref'
RESUlT_BACK_UP_PATH = './语料/rouge结果'


def get_rouge(path):
    hyp_names = sorted(os.listdir(path), key=lambda x: int(x[x.find('_') + 1: x.find('.txt')]))
    ref_names = sorted(os.listdir(ref_path), key=lambda x: int(x[x.find('_') + 1: x.find('.txt')]))
    all_scores = []
    for hyp_name, ref_name in zip(hyp_names, ref_names):
        files_rouge = FilesRouge(os.path.join(path, hyp_name), os.path.join(ref_path, ref_name))
        scores = files_rouge.get_scores()
        avg_score = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}
        for x in avg_score:
            avg_score[x] = {'p': sum([score[x]['p'] for score in scores]) / len(scores),
                            'r': sum([score[x]['r'] for score in scores]) / len(scores),
                            'f': sum([score[x]['f'] for score in scores]) / len(scores)}
        all_scores.append(avg_score)
    return all_scores


def get_avg_rouge(rouge_result):
    avg_rouge = {'rouge-1': 0, 'rouge-l': 0, 'rouge-2': 0}
    for r in rouge_result:
        for t in r:
            avg_rouge[t] += r[t]['f']
    for t in avg_rouge:
        avg_rouge[t] /= len(rouge_result)
    return avg_rouge


def get_avg_rouge_prf(rouge_result):
    avg_rouge = {'rouge-1': {'p': 0, 'r': 0, 'f': 0}, 'rouge-l': {'p': 0, 'r': 0, 'f': 0}, 'rouge-2': {'p': 0, 'r': 0, 'f': 0}}
    for r in rouge_result:
        for t in r:
            avg_rouge[t]['f'] += r[t]['f']
            avg_rouge[t]['p'] += r[t]['p']
            avg_rouge[t]['r'] += r[t]['r']
    for t in avg_rouge:
        avg_rouge[t]['r'] /= len(rouge_result)
        avg_rouge[t]['f'] /= len(rouge_result)
        avg_rouge[t]['p'] /= len(rouge_result)
    return avg_rouge


def plot_baseline_bar():
    hyp_result = get_rouge(hyp_path)
    baseline_result = get_rouge(baseline_path)
    hyp_avg = get_avg_rouge(hyp_result)
    baseline_avg = get_avg_rouge(baseline_result)
    plot.compete_bar(baseline_avg, hyp_avg)
    print(hyp_avg, baseline_avg)


def plot_bar(name):
    rouge_result = get_rouge(hyp_path)
    choose_rouge = [rouge_result[i] for i in range(0, len(rouge_result), 2)]
    recalls_rouge = {'rouge-1': [], 'rouge-l': [], 'rouge-2': []}
    for r in choose_rouge:
        for x in r:
            recalls_rouge[x].append(r[x]['f'])
    plot.draw_bar(name, recalls_rouge[name])


