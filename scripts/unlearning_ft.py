#!/usr/bin/python


import argparse, json
import os
import time
import torch
import copy
import sys

from imageio import imsave

from sg2i.model import SG2I, get_cropped_objs
from sg2i.utils import int_tuple, bool_flag, seed_everthing
from sg2i.vis import draw_scene_graph



# Unlearning methods
from unlearning_method.all_unlearner import ALL_Unlearner # IF
# from unlearning_method.retrain_unlearner import Retrain_Unlearner # Retrain

import cv2
import numpy as np

from sg2i.loader_utils import build_eval_loader
from scripts.eval_utils import save_images, visualize_scene_graphs, custom_grid_sample, compare_models, compare_complex_outputs



from sg2i.data import imagenet_deprocess_batch, imagenet_deprocess_save_batch

from sg2i.loader_utils import build_train_loaders
from scripts.train_utils import *
from setproctitle import setproctitle

import torch.nn.functional as F
from torch.autograd import grad

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms
from PIL import Image



# Metrics
from skimage.metrics import structural_similarity as ssim
from PerceptualSimilarity import models
from scripts.unlearning_utils import SSIMLoss, count_changed_pixels, config_args
import torchvision.transforms as transforms

os.chdir(os.path.dirname(os.path.abspath(__file__)))  #  Change directory to the directory of a Python script


parser = argparse.ArgumentParser()

########################## Unlearning parameters ###########################
parser.add_argument('--unlearn_feature_type', default="visual", type=str, help='visual, obj, bbox, or their combos like: visual_obj')

parser.add_argument('--candidate', type=int, default=0, help='the index of human img to be unleanred')

parser.add_argument('--unlearning_checkpoint', default=None)


parser.add_argument('--finetune', default=False, type=bool_flag)

parser.add_argument('--specific_obj', default=3, type=int)
parser.add_argument('--data_lot_idx', default=0, type=int)
parser.add_argument('--obj_size_threshold', default=0.4, type=float)


parser.add_argument('--unlearn_obj_and_idx', default=None, help='for visualizing scene graphs, (obj, obj_idx)')

'''
  A object idx vocabulary for "human beings" in the vg dataset:
  "man": 3
  "person": 6
  "woman": 20
  "people": 27
  "face": 48
  "boy": 58
  "girl": 75
  "child": 165
  "lady": 170
  human_objs_list = [3, 6, 20, 27, 48, 58, 75, 165, 170]
'''

########################## GIF parameters ###########################
parser.add_argument('--iteration', type=int, default=1)
parser.add_argument('--scale', type=int, default=1000)
parser.add_argument('--damp', type=float, default=0)
parser.add_argument('--unlearn_module', default="gnn", type=str)



parser.add_argument('--deprocess_in_loss', default=False, type=bool_flag)


########################## PATH ###########################

parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--exp_dir', default='experiments/vg/')
parser.add_argument('--result_dir', default='results/spade_64_vg/')
parser.add_argument('--experiment', default="spade_64_vg", type=str)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--predgraphs', default=False, type=bool_flag)
parser.add_argument('--image_size', default=(64, 64), type=int_tuple)
parser.add_argument('--save_images', default=True, type=bool_flag)
parser.add_argument('--image_save_path', default=None, type=str)


parser.add_argument('--generative', default=True, type=bool_flag, 
                    help='if false, reserve the non-roi parts')

########################## Training parameters ###########################
parser.add_argument('--trainset_size', default=200, type=int)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_iterations', default=300000, type=int)
parser.add_argument('--learning_rate', default=2e-4, type=float)
parser.add_argument('--shuffle', default=False, type=bool_flag)
                            
parser.add_argument('--save_graph_image', default=False, type=bool_flag)
parser.add_argument('--save_graph_json', default=False, type=bool_flag) # before and after
parser.add_argument('--with_query_image', default=False, type=bool)
parser.add_argument('--mode', default='remove',
                    choices=['auto_withfeats', 'auto_nofeats', 'replace', 'reposition', 'remove'])
# fancy image save that also visualizes a text label describing the change
parser.add_argument('--label_saved_image', default=True, type=bool_flag)
# used for relationship changes (reposition)
# do we drop the subject, object, or both from the original location?
parser.add_argument('--drop_obj', default=False, type=bool_flag)
parser.add_argument('--drop_subj', default=True, type=bool_flag)
# used for object replacement
# if True, the position of the original object is kept (gt) while the size (H, W) comes from the predicted box
# recommended to set to True when replacing objects (e.g. bus to car),
# and to False for background change (e.g. ocean to field)
parser.add_argument('--combined_gt_pred_box', default=True, type=bool_flag)
# use with mode auto_nofeats to generate diverse objects when features phi are masked/dropped
parser.add_argument('--random_feats', default=False, type=bool_flag, help='if true: image synthesis; if false: image reconstruction')

# Dataset
parser.add_argument('--vg_image_dir', default=os.path.join(DATA_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(DATA_DIR, 'train.h5'))
parser.add_argument('--val_h5', default=os.path.join(DATA_DIR, 'val.h5'))
parser.add_argument('--test_h5', default=os.path.join(DATA_DIR, 'test.h5'))
parser.add_argument('--vocab_json', default=os.path.join(DATA_DIR, 'vocab.json'))
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)

parser.add_argument('--shuffle_train_loader', default=False, type=bool_flag)  # When learning, it is set to true

parser.add_argument('--ablation_mode', default=None,
                    choices=['scalar', 'module'])


# Generic discriminator options
parser.add_argument('--discriminator_loss_weight', default=0.01, type=float)
parser.add_argument('--gan_loss_type', default='gan')
parser.add_argument('--d_normalization', default='batch')
parser.add_argument('--d_padding', default='valid')
parser.add_argument('--d_activation', default='leakyrelu-0.2')

# Object discriminator
parser.add_argument('--d_obj_arch',
    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--crop_size', default=32, type=int)
parser.add_argument('--d_obj_weight', default=1.0, type=float) # multiplied by d_loss_weight #was 1.0
parser.add_argument('--ac_loss_weight', default=0.1, type=float) #was 0.1

# Visualization
parser.add_argument('--methods_involved_list', default=None)

seed_everthing(42)



SPLIT = "test"
parser.add_argument('--data_h5', default=os.path.join(VG_DIR, SPLIT + '.h5'))
parser.add_argument('--data_image_dir',
        default=os.path.join(VG_DIR, 'images'))

args = parser.parse_args()
args.dataset = "vg"



########################## PATH ###########################
if args.predgraphs and SPLIT == "test":
    SPLIT = 'test_predgraphs'
    args.data_h5 = os.path.join(VG_DIR, SPLIT + '.h5')

if args.checkpoint is None:
  args.checkpoint = os.path.join(os.path.abspath('..'), args.exp_dir, args.experiment + f"_model_ft_obj_{args.specific_obj}_{args.data_lot_idx}.pt")



if args.unlearning_checkpoint is None:
  args.unlearning_checkpoint = os.path.join(os.path.abspath('..'), args.exp_dir, "unlearning", args.experiment, 'new_exp', f"model_ft_obj_{args.specific_obj}_{args.data_lot_idx}.pt")
  unlearn_model_checkpoint_path = os.path.join(os.path.abspath('..'), args.exp_dir, "unlearning", args.experiment, 'new_exp')
  if not os.path.exists(unlearn_model_checkpoint_path):
     os.makedirs(unlearn_model_checkpoint_path)

if args.random_feats:
    image_task = 'image_synthesis'
else:
    image_task = 'image_reconstruction'

RESULTS_SAVE_PATH = '/home/ousg/' + args.result_dir + image_task + '/'
if not os.path.exists(RESULTS_SAVE_PATH):
     os.makedirs(RESULTS_SAVE_PATH)
TXT_SAVE_PATH = RESULTS_SAVE_PATH + f'{args.specific_obj}_{args.data_lot_idx}.txt'
LOG_SAVE_PATH = RESULTS_SAVE_PATH + f'{args.specific_obj}_{args.data_lot_idx}.json'
# FEATURE_SAVE_PATH = RESULTS_SAVE_PATH + f'{args.specific_obj}_{args.data_lot_idx}.npy'

IMG_SAVE_PATH = '/home/ousg/' + args.exp_dir + 'logs/{}/evaluation/'.format(args.experiment)

args.image_save_path = IMG_SAVE_PATH

########################## Functions ###########################
def build_model(args, checkpoint):
  model = SG2I(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  # model.eval()
  model.image_size = args.image_size
  model.cuda()
  return model



def process_batch(imgs, imgs_in, objs, boxes, triples, obj_to_img, triples_to_img, device,
                  use_feats=True, filter_box=False):
    num_imgs = imgs.shape[0]
    imgs_stack = []
    imgs_in_stack = []
    boxes_stack = []
    objs_stack = []
    triples_stack = []
    obj_to_img_new = []
    candidates_stack = []
    previous_idx = 0

    for i in range(num_imgs):
        start_idx_for_img = (obj_to_img == i).nonzero()[0]
        last_idx_for_img = (obj_to_img == i).nonzero()[-1]
        boxes_i = boxes[start_idx_for_img: last_idx_for_img + 1, :]     # this includes the 'image' box!
        objs_i = objs[start_idx_for_img: last_idx_for_img + 1]

        start_idx_for_img = (triples_to_img == i).nonzero()[0]
        last_idx_for_img = (triples_to_img == i).nonzero()[-1]
        triples_i = triples[start_idx_for_img:last_idx_for_img + 1]

        num_boxes = boxes_i.shape[0]  # number of boxes in current image minus the 'image' box

        if filter_box:
            min_dim = 0.05  # about 3 pixels
            keep = [b for b in range(boxes_i.shape[0] - 1) if
                    boxes_i[b, 2] - boxes_i[b, 0] > min_dim and boxes_i[b, 3] - boxes_i[b, 1] > min_dim]

            times_to_rep = len(keep)
            img_indices = torch.LongTensor(keep)
        else:

            times_to_rep = num_boxes - 1
            img_indices = torch.arange(0, times_to_rep)
            keep = img_indices

        # boxes that will be dropped for each sample
        drop_indices = torch.zeros_like(img_indices)
        for j in range(len(keep)):
            drop_indices[j] = num_boxes * j + keep[j]

        # replicate things for current image
        imgs_stack.append(imgs[i, :, :, :].repeat(times_to_rep, 1, 1, 1))
        imgs_in_stack.append(imgs_in[i, :, :, :].repeat(times_to_rep, 1, 1, 1))
        objs_stack.append(objs_i.repeat(times_to_rep))     # replicate object ids #boxes times
        boxes_stack.append(boxes_i.repeat(times_to_rep, 1))   # replicate boxes #boxes times

        obj_to_img_new.append(torch.arange(0, times_to_rep).repeat(num_boxes, 1)
                              .transpose(1,0).reshape(-1) + previous_idx)

        previous_idx = obj_to_img_new[-1].max() + 1

        triplet_offsets = num_boxes * torch.arange(0, times_to_rep).repeat(triples_i.size(0), 1)\
            .transpose(1,0).reshape(-1).to(device)

        triples_i = triples_i.repeat(times_to_rep, 1)
        triples_i[:, 0] = triples_i[:, 0] + triplet_offsets     # offset for replicated subjects
        triples_i[:, 2] = triples_i[:, 2] + triplet_offsets     # offset for replicated objects
        triples_stack.append(triples_i)

        # create index to drop for each sample
        candidates = torch.ones(boxes_stack[-1].shape[0], device=device)

        candidates[drop_indices] = 0     # set to zero the boxes that should be dropped
        candidates_stack.append(candidates)

    imgs = torch.cat(imgs_stack)
    imgs_in = torch.cat(imgs_in_stack)
    boxes = torch.cat(boxes_stack)
    objs = torch.cat(objs_stack)
    triples = torch.cat(triples_stack)
    obj_to_img_new = torch.cat(obj_to_img_new)
    candidates = torch.cat(candidates_stack).unsqueeze(1)

    if use_feats:
        feature_candidates = torch.ones((candidates.shape[0], 1), device=device)
    else:
        feature_candidates = candidates

    return imgs, imgs_in, objs, boxes, triples, obj_to_img_new, candidates, feature_candidates




########################## Main ###########################
def main(args):
    ALL_PIXEL_NUM = args.image_size[0] * args.image_size[0] * 3
    # Monkey patch
    original_grid_sample = F.grid_sample
    F.grid_sample = custom_grid_sample

    # Configure hyperparameter
    config_args(args)


    checkpoint = torch.load(args.checkpoint)
    print('Loading model from ', args.checkpoint)
    vocab = checkpoint['model_kwargs']['vocab']
    model = build_model(args, checkpoint)

    model.to(args.device)
    model.train()
    

    # Construct unlearned dataset (request)
    '''
    A object idx vocabulary for "human beings" in the vg dataset:
    "man": 3
    "person": 6
    "woman": 20
    "people": 27
    "face": 48
    "boy": 58
    "girl": 75
    "child": 165
    "lady": 170
    human_objs_list = [3, 6, 20, 27, 48, 58, 75, 165, 170]
    '''
    vocab, train_loader, val_loader = build_train_loaders(args)

    count = 0
    selected_unlearning_sample = 0
    sample_list = []
    f_unlearning_list = []

    real_img_idx_cout = 0
    ####################### construct sample unlearning & object unlearning request #######################
    for batch in train_loader:
      '''
        For one batch: tuple, batch, size 7
        - batch[0]imgs: FloatTensor of shape (N, 3, H, W)
        - batch[1] objs: LongTensor of shape (num_objs,) giving categories for all objects NODES 
        - batch[2] boxes: FloatTensor of shape (num_objs, 4) giving boxes for all objects
        - batch[3] triples: FloatTensor of shape (num_triples, 3) giving all triples, where triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple EDGES
        - batch[4] obj_to_img: LongTensor of shape (num_objs,) mapping objects to images; obj_to_img[i] = n means that objs[i] belongs to imgs[n]
        - batch[5] triple_to_img: LongTensor of shape (num_triples,) mapping triples to images; triple_to_img[t] = n means that triples[t] belongs to imgs[n]
        - batch[6] imgs_masked: FloatTensor of shape (N, 4, H, W)  # not used in our unlearning task
      '''
      f_unlearn_sample = copy.deepcopy(batch)
      f_unlearn_sample = tuple([item.to(args.device) for item in f_unlearn_sample])
      obj_indices = [index for index, obj_class in enumerate(batch[1]) if obj_class == args.specific_obj]
      obj = [int(batch[1][idx]) for idx in obj_indices]
      f_unlearn_obj = (obj[0], obj_indices[0])
      f_unlearning_list.append((f_unlearn_sample, f_unlearn_obj))



      # select sample to be unlearned, default the first one
      if count == selected_unlearning_sample:
        unlearn_sample = copy.deepcopy(batch)
        verify_sample = copy.deepcopy(batch)
        s_unlearn_sample = copy.deepcopy(batch)


        obj_indices = [index for index, obj_class in enumerate(batch[1]) if obj_class == args.specific_obj]
        obj = [int(batch[1][idx]) for idx in obj_indices]
        unlearn_obj = (obj[0], obj_indices[0])  # obj_label, obj_index
        

        
        

        args.unlearn_obj_and_idx = unlearn_obj
      sample_list.append(batch)
      count += 1

    o_unlearn_sample = tuple([item.to(args.device) for item in unlearn_sample])
    s_unlearn_sample = tuple([item.to(args.device) for item in unlearn_sample])

    o_unlearning_request = (o_unlearn_sample, unlearn_obj, 'object')
    s_unlearning_request = (s_unlearn_sample, 'sample')
    f_unlearning_request = (f_unlearning_list, 'feature')


    ####################### construct feature unlearning request #######################
    args.batch_size = 4
    vocab, train_loader, val_loader = build_train_loaders(args)
    for batch in train_loader:
      all_data = copy.deepcopy(batch)
    all_data = tuple([item.to(args.device) for item in all_data])




    ####################### construct feature unlearning request #######################


    # For debugging the nn.Embedding issues
    # initial_weights = model.obj_embeddings.weight.clone().detach()


    # Select unlearning algorithm (initialize unlearner)
    unlearning_checkpoint = {'model_state': None}
    
    unlearner = ALL_Unlearner(args)

    
    # Execute Unlearning
    print('start unlearning ft ...')

    ###################### construct metrics  ######################
    metrics = ['ssim', 'lpips', 'mae', 'cp']
    persp = ['a1', 'a2', 'a3']
    a3_sample_list = [i for i in range(len(sample_list))]

    all_metrics = {p: {m: 0 for m in metrics} for p in persp}
    all_metrics['a3'] = {sample: {m: 0 for m in metrics} for sample in a3_sample_list}

    ###################### construct unlearning methods  ######################
    all_unlearn_method = ['sample_ft', 'sample_ng', 'feature_if', 'feature_ng', 'feature_mask', 'obj_if', 'obj_ng', 'obj_mask_patch', 'obj_mask_noise']
    args.methods_involved_list = all_unlearn_method
    all_unlearned_model = {method: build_model(args, checkpoint) for method in all_unlearn_method}  # Initialize all unlearned models

    # Load state_dict for each model and store the model itself along with the metrics
    for method, unlearned_model in all_unlearned_model.items():
      state_dict_copy = copy.deepcopy(model.state_dict())
      unlearned_model.load_state_dict(state_dict_copy)

    # Store all things into a dict, 1. model 2. metrics 3. output 4. img_red .....
    all_unlearned_model = {method: {'model': unlearned_model, 
                                    'metrics': copy.deepcopy(all_metrics), 
                                    'output': None, 
                                    'img_pred': None, 
                                    'roi': None,
                                    'resize_roi': None,
                                    'norm_resize_roi': None,
                                    'non_roi': None,
                                    'norm_non_roi': None} for method, unlearned_model in all_unlearned_model.items()}
    
    all_unlearned_model['obj_index'] = args.unlearn_obj_and_idx[1]
    
    ###################### Unlearning ######################
    all_unlearned_model['sample_ft']['model'] = unlearner.unlearn_sample_ft(all_unlearned_model['sample_ft']['model'], s_unlearning_request, all_data=all_data)
    all_unlearned_model['sample_ng']['model'] = unlearner.unlearn_sample_ng(all_unlearned_model['sample_ng']['model'], s_unlearning_request, all_data=all_data)
    all_unlearned_model['feature_ng']['model'] = unlearner.unlearn_feature_ng(all_unlearned_model['feature_ng']['model'], s_unlearning_request, all_data=all_data)
    all_unlearned_model['feature_mask']['model'] = unlearner.unlearn_feature_mask(all_unlearned_model['feature_mask']['model'], s_unlearning_request, all_data=all_data)
    all_unlearned_model['feature_if']['model'] = unlearner.unlearn_feature_if(all_unlearned_model['feature_if']['model'], f_unlearning_request)
    all_unlearned_model['obj_if']['model'] = unlearner.unlearn_obj_if(all_unlearned_model['obj_if']['model'], o_unlearning_request, vocab=vocab)
    all_unlearned_model['obj_ng']['model'] = unlearner.unlearn_obj_ng(all_unlearned_model['obj_ng']['model'], s_unlearning_request, all_data=all_data)
    all_unlearned_model['obj_mask_patch']['model'] = unlearner.unlearn_obj_mask_patch(all_unlearned_model['obj_mask_patch']['model'], s_unlearning_request, all_data=all_data)
    all_unlearned_model['obj_mask_noise']['model'] = unlearner.unlearn_obj_mask_noise(all_unlearned_model['obj_mask_noise']['model'], s_unlearning_request, all_data=all_data)


    ###################### Unlearning verification ######################
    print('start verification ...')
    F.grid_sample = original_grid_sample


    ori_model_feature_list = []
    ori_model_output_feature_list = []
    for method in all_unlearn_method:
      all_unlearned_model[method]['stack_feature'] = []
      all_unlearned_model[method]['stack_output_feature'] = []


    for i in range(len(sample_list)): # For all samples in the batch
      verify_sample = sample_list[i]
      verify_sample = tuple([item.to(args.device) for item in verify_sample])

      imgs, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = verify_sample
      if i == 0:
         all_unlearned_model['obj_num'] = len(objs)




      if not args.generative:
        imgs, imgs_in, objs, boxes, triples, obj_to_img, \
        dropimage_indices, dropfeats_indices = [b.to(args.device) for b in process_batch(
            imgs, imgs_in, objs, boxes, triples, obj_to_img, triple_to_img, args.device,
            use_feats=True, filter_box=False)]

        dropbox_indices = dropimage_indices
      else:
        dropbox_indices = torch.ones_like(objs.unsqueeze(1).float()).to(args.device)
        dropfeats_indices = torch.ones_like(objs.unsqueeze(1).float()).to(args.device)
        dropimage_indices = torch.zeros_like(objs.unsqueeze(1).float()).to(args.device)
      masks = None



      ###################### visualize_images ######################
      all_imgs_to_demo = []

      ###################### visualize_scene_graphs ######################
      visualize_scene_graphs(args, obj_to_img, objs, triples, vocab, args.device, img_idx=i)

      ###################### obtain original ground truth ######################
      img = imagenet_deprocess_batch(imgs).float() # original image
      all_imgs_to_demo.append(img)


      ###################### obtain original models' outputs ######################
      ori_model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks, src_image=imgs_in,
                                    keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, random_feats=args.random_feats, mode='eval')[0]
      ori_img_pred = imagenet_deprocess_batch(ori_model_out).float() # original image
      all_imgs_to_demo.append(ori_img_pred)

      ori_model_feature = model.generate_features(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks, src_image=imgs_in,
                                    keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, random_feats=args.random_feats, mode='eval')
      ori_model_feature_list.append(ori_model_feature.cpu().detach())

      ori_model_output_feature = model.generate_output_features(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks, src_image=imgs_in,
                                    keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, random_feats=args.random_feats, mode='eval')

      ori_model_output_feature_list.append(ori_model_output_feature.cpu().detach())


      

      ###################### obtain unlearned models' outputs ######################
      for method in all_unlearn_method:
        all_unlearned_model[method]['output'] = all_unlearned_model[method]['model'](objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks, src_image=imgs_in,
                                    keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, random_feats=args.random_feats, mode='eval')[0]

        unlearned_model_feature = all_unlearned_model[method]['model'].generate_features(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks, src_image=imgs_in,
                                    keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, random_feats=args.random_feats, mode='eval')
        
        all_unlearned_model[method]['stack_feature'].append(unlearned_model_feature.cpu().detach())

        unlearned_model_output_feature = all_unlearned_model[method]['model'].generate_output_features(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks, src_image=imgs_in,
                                    keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, random_feats=args.random_feats, mode='eval')
        
        all_unlearned_model[method]['stack_output_feature'].append(unlearned_model_output_feature.cpu().detach())
        

        

        all_unlearned_model[method]['img_pred'] = imagenet_deprocess_batch(all_unlearned_model[method]['output']).float()
        all_imgs_to_demo.append(all_unlearned_model[method]['img_pred'])


      
      ############# Save images ##############
      # save images
      if args.save_images:
        img_idx = i
        candidate_ = str(args.candidate)
        box_on = True
        save_images(args, candidate_, 
                    all_imgs_to_demo,
                    img_idx, boxes, box_on=box_on, all_data=f_unlearning_list)



      ############# Assessment preparation ##############
      lpips_model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)  # load a lpips model
      
      target_size = (64, 64) # resizer
      resize_transform = transforms.Resize(target_size) # resizer

      ################## Assessment 1: Targetd Object ##################
      
      if i == args.candidate:
        # unlearn obj error
        # get roi
        img_pred_roi = model.img_get_roi(ori_img_pred, boxes, unlearn_obj).unsqueeze(0)
        obj_roi_pixel_num = (img_pred_roi.shape[2] * img_pred_roi.shape[3] * 3)  # pixel in the roi

        for method in all_unlearn_method:
          all_unlearned_model[method]['roi'] = model.img_get_roi(all_unlearned_model[method]['img_pred'], boxes, unlearn_obj).unsqueeze(0)

        # resize
        resized_img_pred_roi = resize_transform(img_pred_roi)

        for method in all_unlearn_method:
          all_unlearned_model[method]['resize_roi'] = resize_transform(all_unlearned_model[method]['roi'])

        ## A1_SSIM
        for method in all_unlearn_method:
          all_unlearned_model[method]['metrics']['a1']['ssim'] = ssim(
                        (resized_img_pred_roi[0:1, :, :, :] / 255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0),
                        (all_unlearned_model[method]['resize_roi'][0:1, :, :, :] / 255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0),
                        win_size=3,
                        full=False,  # return all scores
                        multichannel= True
                    ) 

        ## A1_LPIPS
        normed_resized_img_pred_roi = resized_img_pred_roi[0:1, :, :, :] / 127.5 - 1
        for method in all_unlearn_method:
          all_unlearned_model[method]['norm_resize_roi'] = all_unlearned_model[method]['resize_roi'][0:1, :, :, :] / 127.5 - 1
          all_unlearned_model[method]['metrics']['a1']['lpips'] = float(lpips_model.forward(all_unlearned_model[method]['norm_resize_roi'], normed_resized_img_pred_roi).detach().cpu().numpy())

        ## A1_MAE
        for method in all_unlearn_method:
          all_unlearned_model[method]['metrics']['a1']['mae'] = torch.mean(torch.abs(all_unlearned_model[method]['roi'] - img_pred_roi)).cpu().item()

        ## A1_Changed Pixel (cp)
        
        for method in all_unlearn_method:
          all_unlearned_model[method]['metrics']['a1']['cp'] = count_changed_pixels(all_unlearned_model[method]['roi'], img_pred_roi) / obj_roi_pixel_num
      
      ################## Assessment 2: Rest part of the target sample ##################
      if i == args.candidate:
        img_pred_non_roi = model.img_get_roi(ori_img_pred, boxes, unlearn_obj, remain=True)
        for method in all_unlearn_method:
          all_unlearned_model[method]['non_roi'] = model.img_get_roi(all_unlearned_model[method]['img_pred'], boxes, unlearn_obj, remain=True)
        
          ## A2_SSIM
          all_unlearned_model[method]['metrics']['a2']['ssim'] = ssim(
                          (img_pred_non_roi[0:1, :, :, :] / 255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0),
                          (all_unlearned_model[method]['non_roi'][0:1, :, :, :] / 255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0),
                          win_size=3,
                          full=False,  # return all scores
                          multichannel= True
                      ) 

          ## A2_LPIPS
          all_unlearned_model[method]['norm_non_roi'] = all_unlearned_model[method]['non_roi'][0:1, :, :, :] / 127.5 - 1
          all_unlearned_model[method]['metrics']['a2']['lpips'] = float(lpips_model.forward(all_unlearned_model[method]['norm_non_roi'], img_pred_non_roi).detach().cpu().numpy())

          ## A2_MAE
          all_unlearned_model[method]['metrics']['a2']['mae'] = torch.mean(torch.abs(all_unlearned_model[method]['non_roi'] - img_pred_non_roi)).cpu().item()

          ## A2_Changed Pixel (cp)
          all_unlearned_model[method]['metrics']['a2']['cp'] = (count_changed_pixels(all_unlearned_model[method]['non_roi'], img_pred_non_roi))  / (ALL_PIXEL_NUM - obj_roi_pixel_num)


      ################## Assessment 3: Other samples ##################
      if not i == args.candidate:
        sample = f_unlearning_list[i][0]
        obj_indices = [index for index, obj_class in enumerate(sample[1]) if obj_class == args.specific_obj]
        obj = [int(sample[1][idx]) for idx in obj_indices]
        same_obj = (obj[0], obj_indices[0])

        # unlearn obj error
        # get roi
        img_pred_roi = model.img_get_roi(ori_img_pred, boxes, same_obj).unsqueeze(0)
        resized_img_pred_roi = resize_transform(img_pred_roi)
        normed_resized_img_pred_roi = resized_img_pred_roi[0:1, :, :, :] / 127.5 - 1
        for method in all_unlearn_method:
          all_unlearned_model[method]['roi'] = model.img_get_roi(all_unlearned_model[method]['img_pred'], boxes, same_obj).unsqueeze(0)

          # resize
          all_unlearned_model[method]['resize_roi'] = resize_transform(all_unlearned_model[method]['roi'])
          
          ## A3_SSIM
          all_unlearned_model[method]['metrics']['a3'][i]['ssim'] = ssim(
                          (resized_img_pred_roi[0:1, :, :, :] / 255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0),
                          (all_unlearned_model[method]['resize_roi'][0:1, :, :, :] / 255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0),
                          win_size=3,
                          full=False,  # return all scores
                          multichannel= True
                      ) 

          ## A3_LPIPS
          all_unlearned_model[method]['norm_resize_roi'] = all_unlearned_model[method]['resize_roi'][0:1, :, :, :] / 127.5 - 1
          all_unlearned_model[method]['metrics']['a3'][i]['lpips'] = float(lpips_model.forward(all_unlearned_model[method]['norm_resize_roi'], normed_resized_img_pred_roi).detach().cpu().numpy())

          ## A3_MAE
          all_unlearned_model[method]['metrics']['a3'][i]['mae'] = torch.mean(torch.abs(all_unlearned_model[method]['roi'] - img_pred_roi)).cpu().item()
        

          ## A3_Changed Pixel (cp)
          all_unlearned_model[method]['metrics']['a3'][i]['cp'] = count_changed_pixels(all_unlearned_model[method]['roi'], img_pred_roi) / obj_roi_pixel_num
          
          print()
        


    ########### calculate average A3 metrics ###############
    for method in all_unlearn_method:
      for metric in all_unlearned_model[method]['metrics']['a3'][0].keys():
        print(method, metric)
        print(all_unlearned_model[method]['metrics']['a3'].keys())
        metric_values = [all_unlearned_model[method]['metrics']['a3'][i][metric] for i in range(len(sample_list))]
        avg_metric_values = sum(metric_values) / (len(metric_values) - 1)
        all_unlearned_model[method]['metrics']['a3'][metric] = avg_metric_values
        print()


    ############## Print results ###############
    for method in all_unlearn_method:
      print('*'*20)
      for ass in all_unlearned_model[method]['metrics'].keys():
        for metric in all_unlearned_model[method]['metrics'][ass].keys():
          result = all_unlearned_model[method]['metrics'][ass][metric]
          if ass == 'a3' and isinstance(metric, int):
            # continue # if this, only average results are shown
            print(f'{method}, {ass}, {metric}, {result}')
          else:
            print(f'{method}, {ass}, {metric}, {result}')
    

    ############## Store log ###############
    ori_model_feature = np.vstack(ori_model_feature_list)
    ori_model_output_feature = np.vstack(ori_model_output_feature_list)
    for method in all_unlearn_method:
       all_unlearned_model[method]['stack_feature'] = np.vstack(all_unlearned_model[method]['stack_feature'])
       all_unlearned_model[method]['stack_feature'] = np.vstack((ori_model_feature, all_unlearned_model[method]['stack_feature']))

       all_unlearned_model[method]['stack_output_feature'] = np.vstack(all_unlearned_model[method]['stack_output_feature'])
       all_unlearned_model[method]['stack_output_feature'] = np.vstack((ori_model_output_feature, all_unlearned_model[method]['stack_output_feature'])) 



      #  assert all_unlearned_model[method]['stack_feature'].shape == (80, 260)

    


    log_dict = all_unlearned_model
    for method in all_unlearn_method:
      keys_to_remove = [key for key in log_dict[method].keys() if key not in ('metrics')]
      np.save(RESULTS_SAVE_PATH+f'{args.specific_obj}_{args.data_lot_idx}_{method}.npy', log_dict[method]['stack_feature'])
      np.save(RESULTS_SAVE_PATH+f'{args.specific_obj}_{args.data_lot_idx}_{method}_output.npy', log_dict[method]['stack_output_feature'])
      for key in keys_to_remove:
        log_dict[method].pop(key)
          
    with open(LOG_SAVE_PATH, 'w') as f:
      json.dump(log_dict, f)

    
  




    
    



if __name__ == '__main__':
  sys.stdout = open(TXT_SAVE_PATH,'wt')
  main(args)
  print('done')