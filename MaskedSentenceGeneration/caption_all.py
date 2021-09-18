import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from scipy.misc import imread, imresize
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_annotation(path):
    ret = {}
    anno = json.load(open(path, 'r'))["annotations"]
    for item in anno:
        ret[item["img"]] = {0: item["content"], 1: item["form"], 2: item["context"]}
    return ret


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3, max_len=100):

    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)      # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out_origin = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out_origin.size(1)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = [[] for _ in range(3)]
    complete_seqs_scores = [[] for _ in range(3)]
    ret_seq = []

    for tpc in range(3):

        # treat beam search as batch size = k
        k = beam_size
        encoder_out = encoder_out_origin.expand(k, num_pixels, encoder_dim)         # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)     # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Start decoding
        step = 1
        h, c = decoder.decoders[tpc].init_hidden_state(encoder_out)

        while True:
            embeddings = decoder.decoders[tpc].embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            awe, alpha = decoder.decoders[tpc].attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            gate = decoder.decoders[tpc].sigmoid(decoder.decoders[tpc].f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            h, c = decoder.decoders[tpc].decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.decoders[tpc].fc(h)    # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs[tpc].extend(seqs[complete_inds].tolist())
                complete_seqs_scores[tpc].extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > max_len:
                break
            step += 1

        i = complete_seqs_scores[tpc].index(max(complete_seqs_scores[tpc]))
        seq = complete_seqs[tpc][i]

        ret_seq.append(seq)

    return ret_seq


def inference(args):
    image_root = '../KnowledgeRetrieval/context_art_classification/Data/SemArt/Images/'

    data_test = load_annotation('annotations/semart_topic_annotated_test.json')
    img_names = list(data_test.keys())

    checkpoint = torch.load(args.model)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    ret = {}
    for img in tqdm(img_names):
        img = os.path.join(image_root, img)
        try:
            # Encode, decode with attention and beam search
            seq_list = caption_image_beam_search(encoder, decoder, img, word_map, args.beam_size, args.max_len)

            temp = {}
            for tpc in range(3):
                words = [rev_word_map[ind] for ind in seq_list[tpc]]
                words = ' '.join(words)
                temp[tpc] = words

            ret[os.path.basename(img)] = temp
        except Exception as e:
            print(e)
            continue
    json.dump(ret, open(args.output_file, 'w'))
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to model')
    parser.add_argument('--word_map', help='path to word map JSON')
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--max_len', default=100, type=int, help='max caption length')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    parser.add_argument('--output_file', help='output file to store the results')
    args = parser.parse_args()

    inference(args)

