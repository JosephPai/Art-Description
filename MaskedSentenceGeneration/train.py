import os
import time
import json
import argparse
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, ParallelDecoder
from datasets import CaptionDatasetPara
from utils import adjust_learning_rate, AverageMeter, save_checkpoint, accuracy, clip_gradient
from nltk.translate.bleu_score import corpus_bleu


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='')
# Data parameters
parser.add_argument('--data_folder', type=str, default='dataset_semart')
parser.add_argument('--data_name', type=str, default='SemArt_5_min_word_freq_120_max_len')
# Model parameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--attention_dim', type=int, default=512)
parser.add_argument('--decoder_dim', type=int, default=512)
# Training parameters
parser.add_argument('--emb_dim', type=int, default=512)
parser.add_argument('--encoder_lr', type=float, default=1e-4)
parser.add_argument('--decoder_lr', type=float, default=4e-4)
parser.add_argument('--fine_tune_encoder', type=bool, default=True)
args = parser.parse_args()

# Data parameters
data_folder = args.data_folder          # folder with data files saved by prepare_dataset.py
data_name = args.data_name              # base name shared by data files

# Model parameters
emb_dim = args.emb_dim           # dimension of word embeddings
attention_dim = args.attention_dim     # dimension of attention linear layers
decoder_dim = args.decoder_dim      # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 100                     # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0    # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = args.batch_size
workers = 1                     # for data-loading; right now, only 1 works with h5py
encoder_lr = args.encoder_lr    # learning rate for encoder if fine-tuning
decoder_lr = args.decoder_lr    # learning rate for decoder
grad_clip = 5.                  # clip gradients at an absolute value of
alpha_c = 1.                    # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.                 # BLEU-4 score right now
print_freq = 20                # print training/validation stats every __ batches
fine_tune_encoder = args.fine_tune_encoder        # fine-tune encoder?
checkpoint = None               # path to checkpoint, None if none
num_topics = 3
exp_name = args.exp_name


def main():

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, \
        data_name, word_map, num_topics, exp_name

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = ParallelDecoder(attention_dim=attention_dim,
                                  embed_dim=emb_dim,
                                  decoder_dim=decoder_dim,
                                  vocab_size=len(word_map),
                                  dropout=dropout,
                                  num_topics=num_topics)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDatasetPara(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDatasetPara(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 15:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                epoch=epoch)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(exp_name, epoch, epochs_since_improvement,
                        encoder, decoder,
                        encoder_optimizer, decoder_optimizer,
                        is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()                             # forward prop. + back prop. time
    data_time = AverageMeter()                              # data loading time
    losses = [AverageMeter() for _ in range(num_topics)]    # loss (per word decoded)
    top5accs = [AverageMeter() for _ in range(num_topics)]  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        output_tuple = decoder(imgs, caps, caplens)
        loss = []

        for t in range(num_topics):
            scores, caps_sorted, decode_lengths, alphas, sort_ind = output_tuple[t]

            # After <start>
            targets = caps_sorted[:, 1:]

            offset = 0
            for offset in range(len(decode_lengths)):
                if decode_lengths[offset] <= 0:
                    break
            decode_lengths = decode_lengths[:offset]
            scores = scores[:offset]
            targets = targets[:offset]

            # Remove timesteps that we didn't decode at, or are pads
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss_t = criterion(scores, targets)

            # Add attention regularization
            loss_t += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            loss.append(loss_t)

            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses[t].update(loss_t.item(), sum(decode_lengths))
            top5accs[t].update(top5, sum(decode_lengths))

        # Back prop for all topics together
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss = sum(loss)
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        batch_time.update(time.time() - start)
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Topic-0-Loss {loss_0.val:.4f} ({loss_0.avg:.4f})'
                  'Topic-0-Top-5 Accuracy {top5_0.val:.3f} ({top5_0.avg:.3f})\t'
                  'Topic-1-Loss {loss_1.val:.4f} ({loss_1.avg:.4f})'
                  'Topic-1-Top-5 Accuracy {top5_1.val:.3f} ({top5_1.avg:.3f})\t'
                  'Topic-2-Loss {loss_2.val:.4f} ({loss_2.avg:.4f})'
                  'Topic-2-Top-5 Accuracy {top5_2.val:.3f} ({top5_2.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                      batch_time=batch_time,
                                                                                      data_time=data_time,
                                                                                      loss_0=losses[0], top5_0=top5accs[0],
                                                                                      loss_1=losses[1], top5_1=top5accs[1],
                                                                                      loss_2=losses[2], top5_2=top5accs[2]))


def validate(val_loader, encoder, decoder, criterion, epoch):

    global num_topics

    decoder.eval()      # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = [AverageMeter() for _ in range(num_topics)]
    top5accs = [AverageMeter() for _ in range(num_topics)]

    start = time.time()

    references = list()  # references (true captions)
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            output_tuple = decoder(imgs, caps, caplens)

            for t in range(num_topics):
                scores, caps_sorted, decode_lengths, alphas, sort_ind = output_tuple[t]

                # After <start>
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                scores_copy = scores.clone()
                scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                # Calculate loss
                loss_t = criterion(scores, targets)

                # Add attention regularization
                loss_t += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # Keep track of metrics
                losses[t].update(loss_t.item(), sum(decode_lengths))
                top5 = accuracy(scores, targets, 5)
                top5accs[t].update(top5, sum(decode_lengths))

                # References
                allcaps_t = allcaps[:, t][sort_ind]  # because images were sorted in the decoder
                for j in range(allcaps_t.shape[0]):
                    img_caps = allcaps_t[j].tolist()
                    img_captions = list(
                        map(lambda c:
                            [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}], img_caps))  # remove <start> and pads
                    references.append(img_captions)

                # Hypotheses
                _, preds = torch.max(scores_copy, dim=2)
                preds = preds.tolist()
                temp_preds = list()
                for j, p in enumerate(preds):
                    temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
                preds = temp_preds
                hypotheses.extend(preds)

                assert len(references) == len(hypotheses)

            batch_time.update(time.time() - start)
            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Topic-0-Loss {loss_0.val:.4f} ({loss_0.avg:.4f})'
                      'Topic-0-Top-5 Accuracy {top5_0.val:.3f} ({top5_0.avg:.3f})\t'
                      'Topic-1-Loss {loss_1.val:.4f} ({loss_1.avg:.4f})'
                      'Topic-1-Top-5 Accuracy {top5_1.val:.3f} ({top5_1.avg:.3f})\t'
                      'Topic-2-Loss {loss_2.val:.4f} ({loss_2.avg:.4f})\t'
                      'Topic-2-Top-5 Accuracy {top5_2.val:.3f} ({top5_2.avg:.3f})'.format(i, len(val_loader),batch_time=batch_time,
                                                                                          loss_0=losses[0], top5_0=top5accs[0],
                                                                                          loss_1=losses[1], top5_1=top5accs[1],
                                                                                          loss_2=losses[2], top5_2=top5accs[2]))

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * Topic-0-LOSS - {loss_0.avg:.3f}, Topic-0-TOP-5 ACCURACY - {top5_0.avg:.3f}, '
            'Topic-1-LOSS - {loss_1.avg:.3f}, Topic-1-TOP-5 ACCURACY - {top5_1.avg:.3f}, '
            'Topic-2-LOSS - {loss_2.avg:.3f}, Topic-2-TOP-5 ACCURACY - {top5_2.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss_0=losses[0], top5_0=top5accs[0],
                loss_1=losses[1], top5_1=top5accs[1],
                loss_2=losses[2], top5_2=top5accs[2],
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
