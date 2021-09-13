import os
import sys
import numpy as np
import argparse
import time
import cv2
import json
from tqdm import tqdm
from scipy.misc import imread
import torch
from torch.autograd import Variable

from faster_rcnn import _init_paths
from context_art_classification.model_mtl import MTL
from context_art_classification.attributes import load_att_class
from context_art_classification.dataloader_mtl import InferDatasetMTL
from faster_rcnn.lib.model.faster_rcnn.resnet import resnet
from faster_rcnn.lib.model.utils.blob import im_list_to_blob
from faster_rcnn.lib.model.utils.config import cfg, cfg_from_file, cfg_from_list
from faster_rcnn.lib.model.rpn.bbox_transform import bbox_transform_inv
from faster_rcnn.lib.model.rpn.bbox_transform import clip_boxes
from faster_rcnn.lib.model.nms.nms_wrapper import nms
from faster_rcnn.lib.model.utils.net_utils import vis_detections

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class Attributes(object):
    def __init__(self):
        self.args_dict = self.get_params()
        self.model = self.init_model()

    def get_params(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--mode', default='test', type=str, help='Mode (train | test)')
        parser.add_argument('--model', default='mtl', type=str,
                            help='Model (mtl | kgm). mlt for multitask learning model. kgm for knowledge graph model.')

        # Directories
        parser.add_argument('--dir_data', default='context_art_classification/Data')

        # Files
        parser.add_argument('--vocab_type', default='type2ind.pckl', help='Type classes file')
        parser.add_argument('--vocab_school', default='school2ind.pckl', help='Author classes file')
        parser.add_argument('--vocab_time', default='time2ind.pckl', help='Timeframe classes file')
        parser.add_argument('--vocab_author', default='author2ind.pckl', help='Author classes file')
        # Test
        parser.add_argument('--workers', default=8, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--model_path',
                            default='context_art_classification/Models/mtl-all/best_model.pth.tar',
                            type=str)
        args_dict, _ = parser.parse_known_args()

        return args_dict

    def init_model(self):
        def dict2list(d):
            l = ['' for _ in range(len(d))]
            for k, v in d.items():
                l[int(v)] = k
            return np.asarray(l)

        # Load classes
        type2idx, school2idx, time2idx, author2idx = load_att_class(self.args_dict)
        num_classes = [len(type2idx), len(school2idx), len(time2idx), len(author2idx)]
        self.idx = {'type': dict2list(type2idx),
                    'school': dict2list(school2idx),
                    'time': dict2list(time2idx),
                    'author': dict2list(author2idx)}

        model = MTL(num_classes)
        model.cuda()

        # Load best model
        print("=> loading checkpoint '{}'".format(self.args_dict.model_path))
        checkpoint = torch.load(self.args_dict.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(self.args_dict.model_path, checkpoint['epoch']))
        return model

    def get_attributes(self, image_paths):
        infer_dataset = InferDatasetMTL(image_paths)
        infer_loader = torch.utils.data.DataLoader(infer_dataset,
                                                   batch_size=self.args_dict.batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=self.args_dict.workers,
                                                   drop_last=False)
        self.model.eval()
        for i, inputs in enumerate(infer_loader):
            # Inputs to Variable type
            input_var = inputs.cuda()

            # Output of the model
            with torch.no_grad():
                # output = self.model(input_var[0])
                output = self.model(input_var)
            _, pred_type = torch.max(output[0], 1)
            _, pred_school = torch.max(output[1], 1)
            _, pred_time = torch.max(output[2], 1)
            _, pred_author = torch.max(output[3], 1)

            # Store outputs
            if i == 0:
                out_type = pred_type.data.cpu().numpy()
                out_school = pred_school.data.cpu().numpy()
                out_time = pred_time.data.cpu().numpy()
                out_author = pred_author.data.cpu().numpy()
            else:
                out_type = np.concatenate((out_type, pred_type.data.cpu().numpy()), axis=0)
                out_school = np.concatenate((out_school, pred_school.data.cpu().numpy()), axis=0)
                out_time = np.concatenate((out_time, pred_time.data.cpu().numpy()), axis=0)
                out_author = np.concatenate((out_author, pred_author.data.cpu().numpy()), axis=0)

        out_type_vocb = self.idx['type'][out_type]
        out_school_vocb = self.idx['school'][out_school]
        out_time_vocb = self.idx['time'][out_time]
        out_author_vocb = self.idx['author'][out_author]

        # return out_type, out_school, out_time, out_author
        return out_type_vocb, out_school_vocb, out_time_vocb, out_author_vocb


class Objects(object):
    def __init__(self):
        self.args = self.get_params()
        self.init_model()
        # print(self.args)

    def get_params(self):
        parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
        parser.add_argument('--cfg', dest='cfg_file',
                            help='optional config file',
                            default='faster_rcnn/cfgs/res101.yml', type=str)
        parser.add_argument('--set', dest='set_cfgs',
                            help='set config keys', default=None,
                            nargs=argparse.REMAINDER)
        parser.add_argument('--cag', dest='class_agnostic',
                            help='whether perform class_agnostic bbox regression',
                            action='store_true')
        parser.add_argument('--checkpoint_path',
                            help='checkpoint to load network',
                            default='faster_rcnn/data/pretrained_model/faster_rcnn_1_10_14657_resnet_coco.pth',
                            type=str)
        parser.add_argument('--vis', dest='vis', default=True,
                            help='visualization mode', type=bool)
        args = parser.parse_args()
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]']

        return args

    def _get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
          im (ndarray): a color image in BGR order
        Returns:
          blob (ndarray): a data blob holding an image pyramid
          im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def init_model(self):
        if self.args.cfg_file is not None:
            cfg_from_file(self.args.cfg_file)
        if self.args.set_cfgs is not None:
            cfg_from_list(self.args.set_cfgs)
        cfg.USE_GPU_NMS = True
        cfg.CUDA = True

        # print('Using config:')
        # pprint.pprint(cfg)
        np.random.seed(cfg.RNG_SEED)

        self.object_classes = np.asarray(['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                                          'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                                          'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                                          'bear','zebra', 'giraffe',
                                          'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                                          'snowboard', 'sports ball',
                                          'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                                          'tennis racket', 'bottle',
                                          'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                                          'sandwich', 'orange',
                                          'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                                          'potted plant', 'bed',
                                          'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                                          'cell phone', 'microwave',
                                          'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                                          'teddy bear', 'hair drier', 'toothbrush'])

        self.fasterRCNN = resnet(self.object_classes, 101,
                                 pretrained=False, class_agnostic=self.args.class_agnostic)
        self.fasterRCNN.create_architecture()
        print("load checkpoint %s" % self.args.checkpoint_path)
        checkpoint = torch.load(self.args.checkpoint_path)
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print('load model successfully!')
        self.fasterRCNN.cuda()

    def get_object(self, image_paths, write_result_img=False):
        # initilize the tensor holder here.
        im_data = torch.FloatTensor(1).cuda()
        im_info = torch.FloatTensor(1).cuda()
        num_boxes = torch.LongTensor(1).cuda()
        gt_boxes = torch.FloatTensor(1).cuda()

        # make variable
        with torch.no_grad():
            im_data = Variable(im_data)
            im_info = Variable(im_info)
            num_boxes = Variable(num_boxes)
            gt_boxes = Variable(gt_boxes)

        self.fasterRCNN.eval()
        vis = self.args.vis
        thresh = 0.35
        num_images = len(image_paths)
        # print('Loaded Photo: {} images.'.format(num_images))
        objects_info = [[] for _ in range(num_images)]

        for idx in range(num_images):
            im_in = np.array(imread(image_paths[idx]))
            if len(im_in.shape) == 2:
                im_in = im_in[:, :, np.newaxis]
                im_in = np.concatenate((im_in, im_in, im_in), axis=2)
            # rgb -> bgr
            im_in = im_in[:, :, ::-1]
            im = im_in

            blobs, im_scales = self._get_image_blob(im)
            assert len(im_scales) == 1, "Only single-image batch implemented"
            im_blob = blobs
            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

            im_data_pt = torch.from_numpy(im_blob)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)

            im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.data.resize_(1, 1, 5).zero_()
            num_boxes.data.resize_(1).zero_()

            det_tic = time.time()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = self.fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if self.args.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4 * len(self.object_classes))
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
                pred_boxes = _.cuda()

            pred_boxes /= im_scales[0]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic

            misc_tic = time.time()

            im2show = None
            if vis:
                im2show = np.copy(im)
            for j in xrange(1, len(self.object_classes)):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if self.args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if vis:
                        im2show = vis_detections(im2show, self.object_classes[j], cls_dets.cpu().numpy(), 0.5)
                    objects_info[idx].append(self.object_classes[j])
            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                             .format(num_images + 1, len(image_paths), detect_time, nms_time))
            sys.stdout.flush()

            if write_result_img:
                result_path = os.path.join('faster_rcnn/test_dir/',
                                           os.path.basename(image_paths[idx])[:-4] + "_det.jpg")
                cv2.imwrite(result_path, im2show)
        print(str(objects_info)[2:-2])
        return objects_info


if __name__ == '__main__':
    for phase in ['train', 'test']:
        img_root_dir = 'context_art_classification/Data/SemArt/Images/'

        data = json.load(open('../MaskedSentenceGeneration/data/annotated_{}.json'.format(phase), 'r'))
        img_names = list(data.keys())

        image_paths = [os.path.join(img_root_dir, x.strip()) for x in img_names]
        num = len(image_paths)

        f_words = open('words_{}_set.json'.format(phase), 'w')
        att = Attributes()
        obj = Objects()
        for i in tqdm(range(num)):
            # print("%d/%d: %s" % (i+1, num, image_paths[i]))
            out_type_vocb, out_school_vocb, out_time_vocb, out_author_vocb = att.get_attributes([image_paths[i]])
            object_info = obj.get_object([image_paths[i]])
            result = {'Id': os.path.basename(image_paths[i]),
                      'Type': out_type_vocb[0],
                      'School': out_school_vocb[0],
                      'Time': out_time_vocb[0],
                      'Author': out_author_vocb[0],
                      'Objects': str(object_info)[2:-2]}
            f_words.write(json.dumps(result)+'\n')
        f_words.close()

