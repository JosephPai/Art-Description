import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import h5py
import json
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter, defaultdict
from random import seed, sample
import pandas as pd
from stanfordcorenlp import StanfordCoreNLP
from utils import sentence_normalize, un_capitalize, merge_entities, check_exact_author, list2long


nlp = StanfordCoreNLP('/data00/user1/data/corenlp/')
num_topics = 3


def load_annotation(path):
    ret = {}
    anno = json.load(open(path, 'r'))["annotations"]
    for item in anno:
        ret[item["img"]] = {0: item["content"], 1: item["form"], 2: item["context"]}
    return ret


def tokenize(imgs, max_len=200):
    bad_cnt = 0
    for img in tqdm(imgs):
        author = img['author']
        sent_topic_dict = img['sentences']
        token_topic_dict, template_topic_dict = defaultdict(list), defaultdict(list)

        for topic_key in sent_topic_dict.keys():
            sent_tuple = sent_topic_dict[topic_key]
            token_tuple, template_tuple = [], []
            topic = int(topic_key)

            for sent in sent_tuple:
                sent = sentence_normalize(sent)
                doc = nlp.ner(sent)
                tokens = [d[0] for d in doc]
                template = [d[1] + '_' if d[1] != 'O' else d[0] for d in doc]
                try:
                    tokens, template = un_capitalize(tokens, template)
                    assert len(tokens) == len(template), 'length mismatch after normalize %s' % sent
                    assert len(tokens) > 0, "no tokens in the sentence? %s" % sent

                    template = check_exact_author(tokens, template, author, verbose=False)
                    template = merge_entities(template)
                    assert len(template) > 0, "no tokens after merge entities? %s" % sent
                except Exception as e:
                    print(e)
                    bad_cnt += 1
                    print('%d-th bad sentence %s, just skip it' % (bad_cnt, sent))
                    continue
                token_tuple.append(tokens)
                template_tuple.append(template)

            long_token = list2long(token_tuple, max_len)
            long_template = list2long(template_tuple, max_len)
            token_topic_dict[topic] = long_token
            template_topic_dict[topic] = long_template

        img['sentences'] = [{'long_token': token_topic_dict, 'long_template': template_topic_dict}]

    return imgs


def create_dataset(image_folder, output_folder, max_len=100, min_len=5, min_word_freq=20):

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    df_train = pd.read_csv('../KnowledgeRetrieval/context_art_classification/Data/SemArt/semart_train.csv',
                           delimiter='\t', encoding='Cp1252')
    df_val = pd.read_csv('../KnowledgeRetrieval/context_art_classification/Data/SemArt/semart_val.csv',
                         delimiter='\t', encoding='Cp1252')
    df_test = pd.read_csv('../KnowledgeRetrieval/context_art_classification/Data/SemArt/semart_test.csv',
                          delimiter='\t', encoding='Cp1252')

    img_names = list(df_train['IMAGE_FILE']) + list(df_val['IMAGE_FILE']) + list(df_test['IMAGE_FILE'])
    titles = list(df_train['TITLE']) + list(df_val['TITLE']) + list(df_test['TITLE'])
    types = list(df_train['TYPE']) + list(df_val['TYPE']) + list(df_test['TYPE'])
    schools = list(df_train['SCHOOL']) + list(df_val['SCHOOL']) + list(df_test['SCHOOL'])
    times = list(df_train['TIMEFRAME']) + list(df_val['TIMEFRAME']) + list(df_test['TIMEFRAME'])
    authors = list(df_train['AUTHOR']) + list(df_val['AUTHOR']) + list(df_test['AUTHOR'])
    sents_train = load_annotation('annotations/semart_topic_annotated_train.json')
    sents_test = load_annotation('annotations/semart_topic_annotated_test.json')

    imgs = []

    for i in tqdm(range(len(img_names))):
        jimg = {}
        jimg['id'] = img_names[i].strip().split('.')[0]
        jimg['filename'] = img_names[i]
        jimg['type'] = types[i]
        jimg['school'] = schools[i]
        jimg['timeframe'] = times[i]
        jimg['author'] = authors[i]
        jimg['title'] = titles[i]
        if img_names[i] in sents_train.keys():
            jimg['sentences'] = sents_train[img_names[i]]
            jimg['split'] = 'train'
        elif img_names[i] in sents_test.keys():
            jimg['sentences'] = sents_test[img_names[i]]
            jimg['split'] = 'test'
        else:
            continue

        imgs.append(jimg)

    imgs = tokenize(imgs, max_len=max_len)

    train_image_paths = []
    train_image_captions = []

    test_image_paths = []
    test_image_captions = []

    word_freq = Counter()

    for img in imgs:
        captions = [[] for _ in range(num_topics)]
        for c in img['sentences']:
            for topic in c['long_template'].keys():
                word_freq.update(c['long_template'][topic])
                if min_len < len(c['long_template'][topic]) <= max_len:
                    captions[int(topic)].append(c['long_template'][topic])

        if len(captions[0]) == 0 and len(captions[1]) == 0 and len(captions[2]) == 0:
            continue

        path = os.path.join(image_folder, img['filename'])

        if img['split'] in {'train'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0
    if '<.>' in word_map.keys():
        print('Word map include stop dot <.>')

    base_filename = 'SemArt_{}_min_word_freq_{}_max_len'.format(min_word_freq, max_len)

    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (test_image_paths, test_image_captions, 'TEST')]:
        if len(impaths) == 0:
            continue

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            h.attrs['captions_per_image'] = 1
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')
            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = [[] for _ in range(num_topics)]
            caplens = [[] for _ in range(num_topics)]

            for i, path in enumerate(tqdm(impaths)):
                # Sample captions
                assert len(imcaps[i]) == num_topics
                captions = imcaps[i]
                captions_has_content = [x[0] for x in captions if len(x) != 0]
                for t in range(3):
                    if len(captions[t]) == 0:
                        captions[t] = sample(captions_has_content, k=1)

                # Sanity check
                assert len(captions[0]) > 0 and len(captions[1]) > 0 and len(captions[2]) > 0

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255
                images[i] = img

                for t in range(num_topics):
                    for j, c in enumerate(captions[t]):
                        # Encode captions
                        enc_c = [word_map['<start>']] + \
                                [word_map.get(word, word_map['<unk>']) for word in c] + \
                                [word_map['<end>']] + \
                                [word_map['<pad>']] * (max_len - len(c))

                        # Find caption lengths
                        c_len = len(c) + 2
                        enc_captions[t].append(enc_c)
                        caplens[t].append(c_len)

            # Sanity check
            assert images.shape[0] * 1 == len(enc_captions[0]) == len(caplens[0])
            assert len(enc_captions[0]) == len(enc_captions[1]) == len(enc_captions[2])
            assert len(caplens[0]) == len(caplens[1]) == len(caplens[2])
            print('Total training image: ', images.shape[0])

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


if __name__ == '__main__':
    create_dataset(image_folder='../KnowledgeRetrieval/context_art_classification/Data/SemArt/Images',
                   output_folder='dataset_semart',
                   max_len=120,
                   min_len=10,
                   min_word_freq=5)
    nlp.close()

