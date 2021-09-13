import torch
import unicodedata
import re
import copy
import os


#########################################################################
#                           Training utils
#########################################################################


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(save_name, epoch, epochs_since_improvement,
                    encoder, decoder,
                    encoder_optimizer, decoder_optimizer, is_best):

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}

    filename = 'checkpoint_' + save_name + '.pth.tar'

    if is_best:
        torch.save(state, os.path.join('checkpoints', 'BEST_' + filename))
    else:
        torch.save(state, os.path.join('checkpoints', filename))


def save_checkpoint_epoch(save_name, epoch, epochs_since_improvement,
                          encoder, decoder,
                          encoder_optimizer, decoder_optimizer, is_best):

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}

    if is_best:
        filename = 'BEST_checkpoint_{}.pth.tar'.format(save_name)
    else:
        filename = 'checkpoint_{}_ep_{}.pth.tar'.format(save_name, epoch)
    torch.save(state, os.path.join('checkpoints', filename))


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


def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


#########################################################################
#                           Dataset utils
#########################################################################

def remove_punctuation_for_sentence(sent):
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
                print('abnormal after check author!', templates)
            new_temp.append(w)
    return new_temp


def list2long(tokens, max_len=10000):
    tokens = sorted(tokens, key=lambda x: len(x), reverse=True)
    ret = []
    for t in tokens:
        if len(ret) + len(t) > max_len:
            break
        ret.extend(t)
        ret.extend(['<.>'])
    return ret
