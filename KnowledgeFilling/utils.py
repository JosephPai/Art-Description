import unicodedata
import re
import copy
import json
from collections import defaultdict


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def remove_punctuation_for_sentence(sent):
    # punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    punc = '~`!#$%^&*()_+-=|;":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    new_sent = re.sub(r"[%s]+" % punc, "", sent)
    return new_sent


def remove_non_ascii_for_sentence(sent):
    new_sent = unicodedata.normalize('NFKD', sent).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_sent


def sentence_normalize(sent):
    new_sent = remove_punctuation_for_sentence(sent)
    new_sent = remove_non_ascii_for_sentence(new_sent)
    return new_sent


def un_capitalize(tokens, templates):
    if not templates[0].endswith('_'):
        word = tokens[0]
        word = word.lower()
        tokens[0] = word
        templates[0] = word
    return tokens, templates


def check_exact_author(tokens, templates, author, verbose=False):
    author = author.lower()
    author = author.split(',')
    author = [x.strip() for x in author if len(x) > 2]
    candidate = copy.copy(author)
    candidate.append(' '.join(author))
    candidate.append(' '.join(list(reversed(author))))

    i = 0
    while True:
        if i < len(tokens):
            word = []
            j = 1
            if templates[i] == 'PERSON_':
                word.append(tokens[i].lower())
                while j < 100:
                    if i + j < len(tokens) and templates[i + j] == 'PERSON_':
                        word.append(tokens[i + j].lower())
                        j += 1
                    else:
                        break
                word = ' '.join(word)

                if word in candidate:
                    if verbose:
                        print('check author: find {} in {}, '
                              'with origin sentence: {}.'.format(word, candidate, ' '.join(tokens)))
                    for k in range(j):
                        assert templates[i + k] == 'PERSON_', templates[i + k]
                        templates[i + k] = 'AUTHOR_'
                    assert i + j >= len(tokens) or templates[i + j] != 'PERSON_'
                    break
            i += j
        else:
            break

    return templates


def merge_entities(templates):
    new_temp = [templates[0]]
    for i in range(1, len(templates)):
        prev_w = new_temp[-1]
        w = templates[i]
        if w.endswith('_') and w == prev_w:
            pass
        else:
            if (w == 'AUTHOR_' and prev_w == 'PERSON_') or (prev_w == 'AUTHOR_' and w == 'PERSON_'):
                print('abnormal after check author!!!!!!!!!!!!!!!!!', templates)
            new_temp.append(w)
    return new_temp


def split_long_sent(imgs, min_len):
    counts = {}
    for img in imgs:
        sent = img['sentences']
        sent = sent.strip().split('.')
        sent = list(filter(lambda x: x is not None and len(x) > min_len, sent))
        sent_len = len(sent)
        img['sentences'] = sent
        counts[sent_len] = counts.get(sent_len, 0) + 1

    print('sentence length distribution (count number of SENTENCE!):')
    max_len = max(counts.keys())
    print('max sentence numbers in raw data: ', max_len)
    sum_len = sum(counts.values())
    for i in range(max_len+1):
        print('%2d: %10d   %f%%' % (i, counts.get(i, 0), counts.get(i, 0)*100.0/sum_len))

    return imgs


def get_entities(tokens, templates):
    tokens.append('<end>')
    templates.append('<end>')
    ret = defaultdict(list)
    entity = []
    for i in range(0, len(tokens)-1):
        if tokens[i] != templates[i]:
            entity.append(tokens[i])
            if templates[i] != templates[i+1]:
                new_entity = ' '.join(entity)
                if new_entity not in ret[templates[i]]:
                    ret[templates[i]].append(new_entity)
                entity = []

    tokens = tokens[:-1]
    templates = templates[:-1]

    return ret

