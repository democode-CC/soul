import logging
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sg2i.model import SG2I

from sg2i.discriminators import PatchDiscriminator, AcCropDiscriminator, MultiscaleDiscriminator, divide_pred

torch.cuda.empty_cache()
# import torch_geometric.transforms as T
# from torch_geometric.datasets import Planetoid
# from torch_geometric.data import NeighborSampler
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np

import copy

from unlearning_method.unlearner import Unlearner

from scripts.eval_utils import bbox_coordinates_with_margin, parse_bool, visualize_imgs_boxes, visualize_scene_graphs

# Metrics
from skimage.metrics import structural_similarity as ssim
from PerceptualSimilarity import models

from scripts.unlearning_utils import SSIMLoss, count_changed_pixels



class ALL_Unlearner(Unlearner):
    def __init__(self, args):
        super(ALL_Unlearner, self).__init__(args)
        self.args = args
        self.loss_ssim = SSIMLoss()
    

    def unlearn_obj_if(self, model, unlearning_request, all_data=None, vocab=None):
        '''
        this function is only for object unlearning and sample unlearning
        all_data is for sample unlearning
        '''
        # torch.autograd.set_detect_anomaly(True)
        
        # model = copy.deepcopy(model)

        

        model.train()
        grad_all, grad1, grad2 = None, None, None

        mode = 'eval'

        if unlearning_request[-1] == 'object':
            (unlearn_sample, unlearn_obj, _) = unlearning_request

            ############ load a discriminator ############ 
            obj_discriminator, d_obj_kwargs = build_obj_discriminator(self.args, vocab)
            obj_discriminator.to(self.args.device)
            obj_discriminator.train()
            optimizer_d_obj = torch.optim.Adam(obj_discriminator.parameters(),
                                            lr=self.args.learning_rate)
            
            checkpoint_path = 'spade_64_vg_model.pt'
            checkpoint = torch.load(checkpoint_path)
            obj_discriminator.load_state_dict(checkpoint['d_obj_state'])
            optimizer_d_obj.load_state_dict(checkpoint['d_obj_optim_state'])



            if self.args.dataset == "vg" or (self.args.dataset == "clevr" and not self.args.is_supervised):
                imgs, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = unlearn_sample
            elif self.args.dataset == "clevr":  
                imgs, imgs_src, objs, objs_src, boxes, boxes_src, triples, triples_src, obj_to_img, \
                triple_to_img, imgs_in = unlearn_sample

            objs_ = objs[unlearn_obj[1]].unsqueeze(0)
            boxes_ = boxes[unlearn_obj[1]].unsqueeze(0)
            obj_to_img_ = obj_to_img[unlearn_obj[1]].unsqueeze(0)

            masks = None
            imgs_src = None
            model_boxes = boxes
            model_masks = masks
            dropimage_indices = torch.zeros_like(objs.unsqueeze(1).float()).to(self.args.device)
            dropfeats_indices = torch.ones_like(objs.unsqueeze(1).float()).to(self.args.device)


            # All (minimize loss later) use old obj_vecs
            out_ori = model.forward_feature_unlearn(self.args, objs, triples, obj_to_img,
                            boxes_gt=model_boxes, masks_gt=model_masks, ori_image=imgs, src_image=imgs_in, keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, imgs_src=imgs_src, t=0, mode=mode, 
                            unlearn_obj=unlearn_obj, feature_edit=False, input_object_range='all')  # 
            #  only for loss 3
            out_ori_maksed = model.forward_feature_unlearn(self.args, objs, triples, obj_to_img,
                            boxes_gt=model_boxes, masks_gt=model_masks, ori_image=imgs, src_image=imgs_in, keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, imgs_src=imgs_src, t=0, mode=mode, 
                            unlearn_obj=unlearn_obj, feature_edit=True, input_object_range='all')  # 
            
            
            
            # Remain (maximize loss later) use old obj_vecs
            out_remain = model.forward_feature_unlearn(self.args, objs, triples, obj_to_img,
                            boxes_gt=model_boxes, masks_gt=model_masks, ori_image=imgs, src_image=imgs_in, keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, imgs_src=imgs_src, t=0, mode=mode, 
                            unlearn_obj=unlearn_obj, feature_edit=False, input_object_range='delete')  # 
            
            # Delete (maximize loss later) use new obj_vecs
            out_delete = model.forward_feature_unlearn(self.args, objs, triples, obj_to_img,
                            boxes_gt=model_boxes, masks_gt=model_masks, ori_image=imgs, src_image=imgs_in, keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, imgs_src=imgs_src, t=0, mode=mode, 
                            unlearn_obj=unlearn_obj, feature_edit=True, input_object_range='delete')  # 
            

            

            # Loss design 1: whole image
            loss_ori = self.loss_img(out_ori)
            loss_remain = self.loss_img(out_remain)
            loss_delete = self.loss_img(out_delete)


            # Loss design 2: only on the requested object
            # loss_ori = self.loss_obj(out_ori)
            # loss_delete = self.loss_obj(out_delete)
            # loss_remain = self.loss_obj(out_remain)

            # Loss design 3: use discriminator's loss
            # scores_fake, loss_ori, layers_fake_obj = obj_discriminator(out_ori[1], objs, boxes, obj_to_img)
            # scores_fake, loss_remain, layers_fake_obj = obj_discriminator(out_ori[1], objs_, boxes_, obj_to_img_)
            # scores_fake, loss_delete, layers_fake_obj = obj_discriminator(out_ori_maksed[1], objs_, boxes_, obj_to_img_)


        
        elif unlearning_request[-1] == 'sample':
            (unlearn_sample, _) = unlearning_request


            if self.args.dataset == "vg" or (self.args.dataset == "clevr" and not self.args.is_supervised):
                imgs, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = all_data
            elif self.args.dataset == "clevr":  
                imgs, imgs_src, objs, objs_src, boxes, boxes_src, triples, triples_src, obj_to_img, \
                triple_to_img, imgs_in = all_data

            masks = None
            imgs_src = None
            model_boxes = boxes
            model_masks = masks
            dropimage_indices = torch.zeros_like(objs.unsqueeze(1).float()).to(self.args.device)
            dropfeats_indices = torch.ones_like(objs.unsqueeze(1).float()).to(self.args.device)


            

            # All (minimize loss later) use old obj_vecs
            out_ori = model.forward_sample_unlearn(self.args, objs, triples, obj_to_img,
                            boxes_gt=model_boxes, masks_gt=model_masks, ori_image=imgs, src_image=imgs_in, keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, imgs_src=imgs_src, t=0, mode=mode, pattern='all_all')  # 
            
            
            
            # Remain (maximize loss later) use old obj_vecs
            out_remain = model.forward_sample_unlearn(self.args, objs, triples, obj_to_img,
                            boxes_gt=model_boxes, masks_gt=model_masks, ori_image=imgs, src_image=imgs_in, keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, imgs_src=imgs_src, t=0, mode=mode, pattern='all_del')  # 
            
            if self.args.dataset == "vg" or (self.args.dataset == "clevr" and not self.args.is_supervised):
                imgs, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = unlearn_sample
            elif self.args.dataset == "clevr":  
                imgs, imgs_src, objs, objs_src, boxes, boxes_src, triples, triples_src, obj_to_img, \
                triple_to_img, imgs_in = unlearn_sample

            masks = None
            imgs_src = None
            model_boxes = boxes
            model_masks = masks
            dropimage_indices = torch.zeros_like(objs.unsqueeze(1).float()).to(self.args.device)
            dropfeats_indices = torch.ones_like(objs.unsqueeze(1).float()).to(self.args.device)

            # Delete (maximize loss later) use new obj_vecs
            out_delete = model.forward_sample_unlearn(self.args, objs, triples, obj_to_img,
                            boxes_gt=model_boxes, masks_gt=model_masks, ori_image=imgs, src_image=imgs_in, keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, imgs_src=imgs_src, t=0, mode=mode, pattern='del_del')  # 
        


        


            # loss is the error of between the original object and reconstructed error. Loss = loss_img + loss_obj
            # loss_ori = self.loss_img(out_ori) + self.loss_obj(out_ori)
            # loss_delete = self.loss_img(out_delete) + self.loss_obj(out_delete)
            # loss_remain = self.loss_img(out_remain) + self.loss_obj(out_remain)

            loss_ori = self.loss_img(out_ori)
            loss_remain = self.loss_img(out_remain)
            loss_delete = self.loss_img(out_delete)

        

        ############## Initialize all parameters to False ##############
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        # print(len(list(model.parameters())))
        # print(len(list(model.named_parameters())))

        ############## Select parameters to be edited ##############
        for name, param in model.named_parameters():
            
            # Unlearn all
            if self.args.unlearn_module == 'all':
                if 'obj_emb' in name:
                    param.requires_grad = True
                elif 'pred_emb' in name:
                    param.requires_grad = True
                elif 'gconv' in name:
                    param.requires_grad = True
                elif 'mask' in name:
                    param.requires_grad = True
                elif 'conv_img' in name:
                    param.requires_grad = True
                elif 'decoder' in name:
                    param.requires_grad = True


            # Unlearn Obj_embedder
            if self.args.unlearn_module == 'obj':
                if 'obj_emb' in name:
                    param.requires_grad = True
            
            # Unlearn Pred_embedder
            if self.args.unlearn_module == 'pred':
                if 'pred_emb' in name:
                    param.requires_grad = True


            # Unlearn GNN
            if self.args.unlearn_module == 'gnn':
                if 'gconv.net1.0.weight' in name:
                    param.requires_grad = True
                if 'gconv.net2.0.weight' in name:
                    param.requires_grad = True


            
            # Unlearn Mask
            if self.args.unlearn_module == 'mask':
                if 'mask' in name:
                    param.requires_grad = True
            
            # Unlearn Mask
            if self.args.unlearn_module == 'box':
                if 'box' in name:
                    param.requires_grad = True


            # Unlearn VGG 
            if self.args.unlearn_module == 'high_level':
                if 'high_level' in name:
                    param.requires_grad = True

            # Unlearn CNN branch
            if self.args.unlearn_module == 'cnn':
                if 'conv_img' in name:
                    param.requires_grad = True

            


            # Unlearn Decoder
            if self.args.unlearn_module == 'decoder':
                if 'decoder' in name:
                    param.requires_grad = True
            
            if self.args.unlearn_module == 'decoder_repr':
                if 'decoder_net.decoder' in name:
                    param.requires_grad = True

            if self.args.unlearn_module == 'decoder_output':
                if 'decoder_net.output_conv.2.bias' in name:
                    param.requires_grad = True
            

            # Unlearn norm
            if self.args.unlearn_module == 'norm':
                if 'layer_norm' in name:
                    param.requires_grad = True
            
            



        # # For inspection only
        # non_requires_grad_set = set()
        # for name, param in model.named_parameters():
        #     if not param.requires_grad: # 26 of 219
        #         non_requires_grad_set.add(name)
        #         print(f"Parameter {name} does not require gradients.")


        # indexed_model_parameters_requires_grad = remove_and_reindex(indexed_model_parameters, non_requires_grad_set)


        model_params = [p for p in model.parameters() if p.requires_grad]  # trained model parameter


        grad_all = grad(loss_ori, model_params, retain_graph=True, create_graph=True, allow_unused=True)
        grad1 = grad(loss_delete, model_params, retain_graph=True, create_graph=True, allow_unused=True)  
        grad2 = grad(loss_remain, model_params, retain_graph=True, create_graph=True, allow_unused=True)


        res_tuple = (grad_all, grad1, grad2)  
  

        model = self.gif_approxi(self.args, model, res_tuple, request=unlearning_request[-1])


        return model

    def unlearn_sample_ft(self, model, unlearning_request, all_data=None):
        base, ext = self.args.checkpoint.rsplit('.', 1)
        remain_checkpoint = f"{base}_sample_ft.{ext}"
        checkpoint = torch.load(remain_checkpoint)
        unlearned_model = build_model(self.args, checkpoint)
        unlearned_model.train()
        return unlearned_model
    
    def unlearn_sample_ng(self, model, unlearning_request, all_data=None):
        base, ext = self.args.checkpoint.rsplit('.', 1)
        remain_checkpoint = f"{base}_sample_ng.{ext}"
        checkpoint = torch.load(remain_checkpoint)
        unlearned_model = build_model(self.args, checkpoint)
        unlearned_model.train()
        return unlearned_model
    
    def unlearn_feature_mask(self, model, unlearning_request, all_data=None):
        base, ext = self.args.checkpoint.rsplit('.', 1)
        remain_checkpoint = f"{base}_feature_mask.{ext}"
        checkpoint = torch.load(remain_checkpoint)
        unlearned_model = build_model(self.args, checkpoint)
        unlearned_model.train()
        return unlearned_model

    def unlearn_feature_ng(self, model, unlearning_request, all_data=None):
        base, ext = self.args.checkpoint.rsplit('.', 1)
        remain_checkpoint = f"{base}_feature_ng.{ext}"
        checkpoint = torch.load(remain_checkpoint)
        unlearned_model = build_model(self.args, checkpoint)
        unlearned_model.train()
        return unlearned_model

    def unlearn_obj_mask_patch(self, model, unlearning_request, all_data=None):
        base, ext = self.args.checkpoint.rsplit('.', 1)
        remain_checkpoint = f"{base}_obj_mask_patch.{ext}"
        checkpoint = torch.load(remain_checkpoint)
        unlearned_model = build_model(self.args, checkpoint)
        unlearned_model.train()
        return unlearned_model

    def unlearn_obj_mask_noise(self, model, unlearning_request, all_data=None):
        base, ext = self.args.checkpoint.rsplit('.', 1)
        remain_checkpoint = f"{base}_obj_mask_noise.{ext}"
        checkpoint = torch.load(remain_checkpoint)
        unlearned_model = build_model(self.args, checkpoint)
        unlearned_model.train()
        return unlearned_model

    def unlearn_obj_ng(self, model, unlearning_request, all_data=None):
        base, ext = self.args.checkpoint.rsplit('.', 1)
        remain_checkpoint = f"{base}_obj_ng.{ext}"
        checkpoint = torch.load(remain_checkpoint)
        unlearned_model = build_model(self.args, checkpoint)
        unlearned_model.train()
        return unlearned_model
        

    

    def unlearn_feature_if(self, model, unlearning_request, all_data=None):
        '''
        this function is only for feature unlearning
        all_data is for sample unlearning
        '''
        model.train()
        grad_all, grad1, grad2 = None, None, None
        mode = 'eval'

        assert unlearning_request[-1] == 'feature'

        (f_unlearning_list, _) = unlearning_request

        for i in range(len(f_unlearning_list)):
            '''
            4
            '''
            unlearn_sample = f_unlearning_list[i][0]
            unlearn_obj = f_unlearning_list[i][1]


            if self.args.dataset == "vg" or (self.args.dataset == "clevr" and not self.args.is_supervised):
                imgs, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = unlearn_sample
            elif self.args.dataset == "clevr":  
                imgs, imgs_src, objs, objs_src, boxes, boxes_src, triples, triples_src, obj_to_img, \
                triple_to_img, imgs_in = unlearn_sample

            masks = None
            imgs_src = None
            model_boxes = boxes
            model_masks = masks
            dropimage_indices = torch.zeros_like(objs.unsqueeze(1).float()).to(self.args.device)
            dropfeats_indices = torch.ones_like(objs.unsqueeze(1).float()).to(self.args.device)

            # All (minimize loss later) use old obj_vecs
            out_ori = model.forward_feature_unlearn(self.args, objs, triples, obj_to_img,
                            boxes_gt=model_boxes, masks_gt=model_masks, ori_image=imgs, src_image=imgs_in, keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, imgs_src=imgs_src, t=0, mode=mode, 
                            unlearn_obj=unlearn_obj, feature_edit=False, input_object_range='all')  # 
            
            
            
            # Remain (maximize loss later) use old obj_vecs
            out_remain = model.forward_feature_unlearn(self.args, objs, triples, obj_to_img,
                            boxes_gt=model_boxes, masks_gt=model_masks, ori_image=imgs, src_image=imgs_in, keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, imgs_src=imgs_src, t=0, mode=mode, 
                            unlearn_obj=unlearn_obj, feature_edit=False, input_object_range='delete')  # 
            
            # Delete (maximize loss later) use new obj_vecs
            out_delete = model.forward_feature_unlearn(self.args, objs, triples, obj_to_img,
                            boxes_gt=model_boxes, masks_gt=model_masks, ori_image=imgs, src_image=imgs_in, keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                    keep_image_idx=dropimage_indices, imgs_src=imgs_src, t=0, mode=mode, 
                            unlearn_obj=unlearn_obj, feature_edit=True, input_object_range='delete')  # 
            
            loss_ori = self.loss_obj(out_ori)
            loss_remain = self.loss_obj(out_remain)
            loss_delete = self.loss_obj(out_delete)

            ############## Initialize all parameters to False ##############
            for name, param in model.named_parameters():
                param.requires_grad = False

            # print(len(list(model.parameters())))
            # print(len(list(model.named_parameters())))

            ############## Select parameters to be edited ##############
            for name, param in model.named_parameters():
                
                # Unlearn all
                if self.args.unlearn_module == 'all':
                    if 'obj_emb' in name:
                        param.requires_grad = True
                    elif 'pred_emb' in name:
                        param.requires_grad = True
                    elif 'gconv' in name:
                        param.requires_grad = True
                    elif 'mask' in name:
                        param.requires_grad = True
                    elif 'conv_img' in name:
                        param.requires_grad = True
                    elif 'decoder' in name:
                        param.requires_grad = True


                # Unlearn Obj_embedder
                if self.args.unlearn_module == 'obj':
                    if 'obj_emb' in name:
                        param.requires_grad = True
                
                # Unlearn Pred_embedder
                if self.args.unlearn_module == 'pred':
                    if 'pred_emb' in name:
                        param.requires_grad = True


                # Unlearn GNN
                if self.args.unlearn_module == 'gnn':
                    if 'gconv.net1.0.weight' in name:
                        param.requires_grad = True
                    if 'gconv.net2.0.weight' in name:
                        param.requires_grad = True


                
                # Unlearn Mask
                if self.args.unlearn_module == 'mask':
                    if 'mask' in name:
                        param.requires_grad = True
                
                # Unlearn Mask
                if self.args.unlearn_module == 'box':
                    if 'box' in name:
                        param.requires_grad = True


                # Unlearn VGG 
                if self.args.unlearn_module == 'high_level':
                    if 'high_level' in name:
                        param.requires_grad = True

                # Unlearn CNN branch
                if self.args.unlearn_module == 'cnn':
                    if 'conv_img' in name:
                        param.requires_grad = True

                


                # Unlearn Decoder
                if self.args.unlearn_module == 'decoder':
                    if 'decoder' in name:
                        param.requires_grad = True
                
                if self.args.unlearn_module == 'decoder_repr':
                    if 'decoder_net.decoder' in name:
                        param.requires_grad = True

                if self.args.unlearn_module == 'decoder_output':
                    if 'decoder_net.output_conv.2.bias' in name:
                        param.requires_grad = True
                

                # Unlearn norm
                if self.args.unlearn_module == 'norm':
                    if 'layer_norm' in name:
                        param.requires_grad = True
                

            # # For inspection only
            # non_requires_grad_set = set()
            # for name, param in model.named_parameters():
            #     if not param.requires_grad: # 26 of 219
            #         non_requires_grad_set.add(name)
            #         print(f"Parameter {name} does not require gradients.")


            # indexed_model_parameters_requires_grad = remove_and_reindex(indexed_model_parameters, non_requires_grad_set)


            model_params = [p for p in model.parameters() if p.requires_grad]  # trained model parameter


            grad_all = grad(loss_ori, model_params, retain_graph=True, create_graph=True, allow_unused=True)
            grad1 = grad(loss_delete, model_params, retain_graph=True, create_graph=True, allow_unused=True)  
            grad2 = grad(loss_remain, model_params, retain_graph=True, create_graph=True, allow_unused=True)

            
            res_tuple = (grad_all, grad2, grad1)

            model = self.gif_approxi(self.args, model, res_tuple, request='feature')  #


        
        return model






    
    def loss_img(self, out, metric='ssim'):  # loss function is signficant for influence estimation!!!
        if metric == 'mae':
            # loss = torch.mean(torch.abs(out[0] - out[1]).view(out[0].shape[0], -1), 1).cpu().numpy()
            loss = torch.mean(torch.abs(out[0] - out[1]))
        elif metric == 'ssim':
            loss = self.loss_ssim(out[0], out[1])
        return loss

    
    def loss_obj(self, out, metric='ssim'):  # loss function is signficant for influence estimation!!!
        if metric == 'mae':
            loss = torch.mean(torch.abs(out[2] - out[3]))
        elif metric == 'ssim':
            loss = self.loss_ssim(out[2], out[3])
        return loss




    def gif_approxi(self, args, model, res_tuple, request=None):
        '''
        res_tuple == (grad_all, grad1, grad2)
        '''


        start_time = time.time()
        iteration, damp, scale = args.iteration, args.damp, args.scale

        if request == 'feature':
            iteration = 8
        if request == 'object':
            if args.specific_obj == 75:
                iteration = 10
                scale=100
            elif args.specific_obj == 3:
                iteration = 2
            else:
                iteration = 1


        
        v = res_tuple[1]
        h_estimate = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        for _ in range(iteration):
            model_named_params  = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
            hv            = self.hvps(res_tuple[0], model_named_params, h_estimate)
            with torch.no_grad():
                h_estimate    = [ v1 + (1-damp)*h_estimate1 - hv1/scale
                            for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]

        model_params = [p for (n,p) in model_named_params]
        params_change = [h_est / scale for h_est in h_estimate]
        params_esti   = [p1 + p2 for p1, p2 in zip(params_change, model_params)]

        idx = 0
        for name, p in model.named_parameters():
            if name in [n for n,p in model.named_parameters() if p.requires_grad]:
                p.data = params_esti[idx]
                idx = idx + 1
            else:
                pass
        return model

    def hvps(self, grad_all, model_named_params, h_estimate):
            element_product = 0
            for grad_elem, v_elem in zip(grad_all, h_estimate):
                element_product += torch.sum(grad_elem * v_elem)
            
            model_params = [p for (n,p) in model_named_params]
            return_grads = grad(element_product,model_params,create_graph=True)
            return return_grads
    

def remove_and_reindex(lst, items_to_remove):
    filtered_lst = [item for item in lst if item[1] not in items_to_remove]
    reindexed_lst = [(index, item[1]) for index, item in enumerate(filtered_lst)]
    
    return reindexed_lst

def build_model(args, checkpoint):
  model = SG2I(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
#   model.eval()
  model.image_size = args.image_size
  model.cuda()
  return model

def build_obj_discriminator(args, vocab):
  discriminator = None
  d_kwargs = {}
  d_weight = args.discriminator_loss_weight
  d_obj_weight = args.d_obj_weight
  if d_weight == 0 or d_obj_weight == 0:
    return discriminator, d_kwargs

  d_kwargs = {
    'vocab': vocab,
    'arch': args.d_obj_arch,
    'normalization': args.d_normalization,
    'activation': args.d_activation,
    'padding': args.d_padding,
    'object_size': args.crop_size,
  }
  discriminator = AcCropDiscriminator(**d_kwargs)

  return discriminator, d_kwargs