import os
import json
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from stanfordcorenlp import StanfordCoreNLP
from utils import sentence_normalize, un_capitalize, merge_entities, check_exact_author, get_entities

nlp = StanfordCoreNLP('/data00/user1/data/corenlp/')


def load_annotation(path):
    ret = {}
    anno = json.load(open(path, 'r'))["annotations"]
    for item in anno:
        ret[item["img"]] = {0: item["content"], 1: item["form"], 2: item["context"]}
    return ret


def extract_words_from_knowledge(knowledge, candidate_words):
    topk = 3
    query = str(knowledge['Query']).strip().split(' ')
    if type(query) == list:
        candidate_words['MISC_'] = query
    elif type(query) == str:
        candidate_words['MISC_'].append(query)
    else:
        raise RuntimeError

    for knowledge_doc in knowledge['knowledge'][:topk]:
        text = knowledge_doc['context']['text']

        text = sentence_normalize(text)
        doc = nlp.ner(text)
        tokens = [d[0] for d in doc]
        template = [d[1] + '_' if d[1] != 'O' else d[0] for d in doc]
        entities_in_sent = get_entities(tokens, template)
        for k in entities_in_sent.keys():
            for new_entity in entities_in_sent[k]:
                if new_entity not in candidate_words[k]:
                    candidate_words[k].append(new_entity)

    return candidate_words


def prepare_pair_data(imgs):

    bad_cnt = 0

    for img in tqdm(imgs):
        author = img['author']
        sent_topic_dict = img['sentences']
        token_topic_dict, template_topic_dict, candidate_words = defaultdict(), defaultdict(), defaultdict(list)

        for topic in sent_topic_dict.keys():
            sent_tuple = sent_topic_dict[topic]
            token_tuple, template_tuple = [], []

            for sent in sent_tuple:
                sent = sentence_normalize(sent)
                doc = nlp.ner(sent)
                tokens = [d[0] for d in doc]
                template = [d[1] + '_' if d[1] != 'O' else d[0] for d in doc]

                # add entity words
                candidate_words = extract_words_from_knowledge(img['knowledge'], candidate_words)
                if img['author'] not in candidate_words['PERSON_']:
                    candidate_words['PERSON_'].append(img['author'])
                if img['school'] not in candidate_words['LOCATION_']:
                    candidate_words['LOCATION_'].append(img['school'])
                if img['timeframe'] not in candidate_words['DATE_']:
                    candidate_words['DATE_'].append(img['timeframe'])
                if img['type'] not in candidate_words['MISC_']:
                    candidate_words['MISC_'].append(img['type'])

                # refine pair data
                try:
                    tokens, template = un_capitalize(tokens, template)
                    assert len(tokens) == len(template), 'length mismatch after normalize %s' % sent
                    assert len(tokens) > 0, "no tokens in the sentence? %s" % sent

                    template = check_exact_author(tokens, template, author)
                    template = merge_entities(template)
                    assert len(template) > 0, "no tokens after merge entities? %s" % sent
                except Exception as e:
                    print(e)
                    bad_cnt += 1
                    print('%d-th bad sentence %s, just skip it' % (bad_cnt, sent))
                    continue
                token_tuple.append(' '.join(tokens))
                template_tuple.append(' '.join(template))

            assert len(token_tuple) == len(template_tuple)
            long_token = '. '.join(token_tuple)
            long_template = '. '.join(template_tuple)
            token_topic_dict[topic] = long_token
            template_topic_dict[topic] = long_template

        img['sentences'] = {'tokens': token_topic_dict, 'template_tokens': template_topic_dict}
        img['candidate_words'] = candidate_words

    return imgs


def create_dataset(output_folder):

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    df_train = pd.read_csv('../KnowledgeRetrieval/context_art_classification/Data/SemArt/semart_train.csv',
                           delimiter='\t', encoding='Cp1252')
    df_val = pd.read_csv('../KnowledgeRetrieval/context_art_classification/Data/SemArt/semart_val.csv',
                         delimiter='\t', encoding='Cp1252')
    df_test = pd.read_csv('../KnowledgeRetrieval/context_art_classification/Data/SemArt/semart_test.csv',
                          delimiter='\t', encoding='Cp1252')

    img_names = list(df_train['IMAGE_FILE']) + list(df_val['IMAGE_FILE']) + list(df_test['IMAGE_FILE'])
    types = list(df_train['TYPE']) + list(df_val['TYPE']) + list(df_test['TYPE'])
    schools = list(df_train['SCHOOL']) + list(df_val['SCHOOL']) + list(df_test['SCHOOL'])
    times = list(df_train['TIMEFRAME']) + list(df_val['TIMEFRAME']) + list(df_test['TIMEFRAME'])
    authors = list(df_train['AUTHOR']) + list(df_val['AUTHOR']) + list(df_test['AUTHOR'])
    sents_train = load_annotation('annotations/semart_topic_annotated_train.json')
    sents_test = load_annotation('annotations/semart_topic_annotated_test.json')

    knowledge_train, knowledge_test = {}, {}
    with open('../KnowledgeRetrieval/retrieved_paragraph_train.json', 'r') as f:
        for line in f:
            knowledge_data = json.loads(line)
            knowledge_train[knowledge_data['Id']] = {'Query': knowledge_data['Query'], 'knowledge': knowledge_data['Result']}
    with open('../KnowledgeRetrieval/retrieved_paragraph_test.json', 'r') as f:
        for line in f:
            knowledge_data = json.loads(line)
            knowledge_test[knowledge_data['Id']] = {'Query': knowledge_data['Query'], 'knowledge': knowledge_data['Result']}

    all_data_num = len(img_names)
    imgs = []

    for i in tqdm(range(all_data_num)):
        jimg = {}
        jimg['id'] = img_names[i].strip().split('.')[0]
        jimg['filename'] = img_names[i]
        jimg['type'] = types[i]
        jimg['school'] = schools[i]
        jimg['timeframe'] = times[i]
        jimg['author'] = authors[i]
        if img_names[i] in sents_train.keys():
            jimg['sentences'] = sents_train[img_names[i]]
            jimg['knowledge'] = knowledge_train[jimg['filename']]
            jimg['split'] = 'train'
        elif img_names[i] in sents_test.keys():
            jimg['sentences'] = sents_test[img_names[i]]
            jimg['knowledge'] = knowledge_test[jimg['filename']]
            jimg['split'] = 'test'
        else:
            continue

        imgs.append(jimg)

    imgs = prepare_pair_data(imgs)

    # img_name: topic: [template, candidates, sentences]
    train_data, test_data = {}, {}

    for img in imgs:
        one_img_sample = {0: [], 1: [], 2: []}
        template_dict, token_dict = img['sentences']['template_tokens'], img['sentences']['tokens']
        for topic in template_dict.keys():
            one_train_tuple = [template_dict[topic], img['candidate_words'], token_dict[topic]]
            one_img_sample[int(topic)] = one_train_tuple

        if img['split'] in {'train'}:
            train_data[img['filename']] = one_img_sample
        elif img['split'] in {'test'}:
            test_data[img['filename']] = one_img_sample

    for data, split in [[train_data, 'TRAIN'],
                        [test_data, 'TEST']]:
        with open(os.path.join(output_folder, split + '_data.json'), 'w') as j:
            json.dump(data, j)


if __name__ == '__main__':
    create_dataset(output_folder='drqa_knowledge_filling_dataset')
    nlp.close()

