import torch
import json
from transformers import BertForMaskedLM
from transformers import BertTokenizer
from transformers import PYTORCH_PRETRAINED_BERT_CACHE
import argparse
import os
import random
import sys
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

np.set_printoptions(threshold=sys.maxsize)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument("--do_lower_case", default=True)
    parser.add_argument('--seed', type=int, default=181)
    parser.add_argument("--device", default='cuda', type=str, help="cuda, cpu")
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--train_name", default="BERT-FillingBlank")
    args, unknown = parser.parse_known_args()
    return args


# Class to contain a single instance of the dataset
class DataSample(object):
    def __init__(self, template_sent, original_sent, candidate_words, img_id='', topic=0):
        self.template = template_sent
        self.original = original_sent
        self.candidates = candidate_words
        self.img_id = img_id
        self.topic = topic


# Dataloader class
class InferenceDataset(data.Dataset):

    def __init__(self, args, template_file_path, candidate_file_path, tokenizer):
        # Params
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.template_file_path = template_file_path
        self.candidate_file_path = candidate_file_path

        # load data and preprocess
        self.samples = self.data_load_and_process(args, template_file_path, candidate_file_path)
        self.num_samples = len(self.samples)
        logger.info('Loaded data with %d samples' % self.num_samples)

    def data_load_and_process(self, args, template_file_path, candidate_file_path):
        with open(template_file_path, 'r') as j:
            templates = json.load(j)
        with open(candidate_file_path, 'r') as j:
            val_data = json.load(j)

        samples = []
        for img in templates.keys():

            if not img in val_data.keys():
                continue

            template_dict = templates[img]
            for topic in ['0', '1', '2']:
                if len(val_data[img][topic]) != 0:
                    candidates = val_data[img][topic][1]
                    break
            list_candidates = [word for type in candidates for word in candidates[type]]
            string_candidates = ', '.join(list_candidates)

            for topic in template_dict.keys():
                temp_sent = template_dict[topic]
                if len(val_data[img][str(topic)]) != 0:
                    original = val_data[img][str(topic)][2]
                else:
                    original = 'No expected output for this topic in meta-data.'

                # remove <start>, <end> and <.>
                temp_sent = str(temp_sent).replace('<start>', '')
                temp_sent = temp_sent.replace('<end>', '')
                temp_sent = temp_sent.replace('<.>', '')
                temp_sent = temp_sent.rsplit(' ', 1)[0]
                samples.append(DataSample(template_sent=temp_sent, candidate_words=string_candidates,
                                          original_sent=original, img_id=img, topic=topic))

        return samples

    def __len__(self):
        return self.num_samples

    def truncate(self, template_tokens, candidates_tokens, max_len):
        while True:
            total_length = len(template_tokens) + len(candidates_tokens)
            if total_length <= max_len:
                break
            else:
                candidates_tokens.pop()

    def truncate_target(self, target_tokens, max_len):
        while len(target_tokens) > max_len:
            target_tokens.pop()

    def __getitem__(self, index):
        # Input sequences: [CLS] + template + [SEP] + candidates + [SEP]
        #  Target sequence: [CLS] + original + [SEP]

        sample = self.samples[index]

        # Tokenize each string individually
        template_tokens = self.tokenizer.tokenize(sample.template)
        original_tokens = self.tokenizer.tokenize(sample.original)
        candidates_tokens = self.tokenizer.tokenize(sample.candidates)

        # Truncate "candidate_tokens" (inplace) if longer than max_seq_lenght (leave 3 tokens for [CLS], [SEP] and [SEP])
        self.truncate(template_tokens, candidates_tokens, self.max_seq_length - 3)
        self.truncate_target(original_tokens, self.max_seq_length - 2)

        # Construct input sequence of tokens
        tokens = [self.tokenizer.cls_token] + template_tokens + [self.tokenizer.sep_token] + candidates_tokens + [self.tokenizer.sep_token]
        segment_ids = [0] * (len(template_tokens) + 2) + [1] * (len(candidates_tokens) + 1)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        # Construct output sequence of tokens
        output_tokens = [self.tokenizer.cls_token] + original_tokens + [self.tokenizer.sep_token]
        output_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        output_padding = [-1] * (self.max_seq_length - len(output_ids)) # -1 to ignore tokens that are not in output sequence
        output_ids += output_padding
        assert len(output_ids) == self.max_seq_length

        # Convert to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        output_ids = torch.tensor(output_ids, dtype=torch.long)
        return input_ids, input_mask, segment_ids, output_ids, sample.img_id, int(sample.topic)


def test(args):
    modeldir = os.path.join('TrainingDrQASrc', args.train_name)

    template_file_path = ''
    candidate_file_path = ''
    output_file = ''

    # Prepare GPUs
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(args.device, n_gpu))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load BERT pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model (in our case, BertForMaskedLM should do the job!)
    model = BertForMaskedLM.from_pretrained(modeldir, cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE._str,
                                                                             'distributed_{}'.format(-1)))
    model.to(args.device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Load infer data
    valDataObject = InferenceDataset(args, template_file_path, candidate_file_path, tokenizer=tokenizer)
    val_dataloader = torch.utils.data.DataLoader(valDataObject, batch_size=args.eval_batch_size, shuffle=False,
                                                 pin_memory=True, num_workers=args.workers, drop_last=False)

    output_data = {}
    model.eval()
    for step, batch in enumerate(tqdm(val_dataloader, desc="Val iter")):
        batch = tuple(t.to(args.device) if not isinstance(t, list) else t for t in batch)
        input_ids, input_mask, segment_ids, output_ids, img_names_batch, topics_batch = batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                            masked_lm_labels=output_ids)
            loss, prediction_scores = outputs[:2]
            prediction_ids = torch.argmax(prediction_scores, dim=2, keepdim=False)

            for idx in range(input_ids.size(0)):

                if img_names_batch[idx] not in output_data.keys():
                    output_data[img_names_batch[idx]] = {0: {'BERT_input': '', 'BERT_output': '', 'BERT_target': ''},
                                                         1: {'BERT_input': '', 'BERT_output': '', 'BERT_target': ''},
                                                         2: {'BERT_input': '', 'BERT_output': '', 'BERT_target': ''}}

                # record bert input
                input_id_sample = input_ids[idx].cpu().data.numpy().tolist()
                if 0 in input_id_sample:
                    end_idx = input_id_sample.index(0)
                else:
                    end_idx = -1
                input_id_sample = input_id_sample[:end_idx]
                input_tokens = tokenizer.convert_ids_to_tokens(input_id_sample)
                input_sent = tokenizer.convert_tokens_to_string(input_tokens)
                output_data[img_names_batch[idx]][int(topics_batch[idx])]['BERT_input'] = input_sent

                # record bert output
                pred_ids = prediction_ids[idx].cpu().data.numpy().tolist()
                pred_ids = pred_ids[:end_idx + 20]
                pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids)
                pred_sent = tokenizer.convert_tokens_to_string(pred_tokens)
                output_data[img_names_batch[idx]][int(topics_batch[idx])]['BERT_output'] = pred_sent

                # record bert target
                label_ids = output_ids[idx].cpu().data.numpy().tolist()
                if -1 in label_ids:
                    end_idx = label_ids.index(-1)         # only matters without mask
                else:
                    end_idx = -1
                label_ids = label_ids[:end_idx]
                label_tokens = tokenizer.convert_ids_to_tokens(label_ids)
                label_sent = tokenizer.convert_tokens_to_string(label_tokens)
                output_data[img_names_batch[idx]][int(topics_batch[idx])]['BERT_target'] = label_sent

    print('Writing to file {}'.format(output_file))
    with open(output_file, 'w') as j:
        json.dump(output_data, j)
    print('Done!')


if __name__ == "__main__":
    args = get_params()
    test(args)

