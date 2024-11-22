#!/usr/bin/python


import os
import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2
import torch
from sg2i.vis import draw_scene_graph
from sg2i.data import imagenet_deprocess_batch
from imageio import imsave


from sg2i.model import mask_image_in_bbox


def remove_node(objs, triples, boxes, imgs, idx, obj_to_img, triple_to_img):
  '''
  removes nodes and all related edges in case of object removal
  image is also masked in the respective area
  idx: list of object ids to be removed
  Returns:
    updated objs, triples, boxes, imgs, obj_to_img, triple_to_img
  '''

  # object nodes
  idlist = list(range(objs.shape[0]))
  keeps = [i for i in idlist if i not in idx]
  objs_reduced = objs[keeps]
  boxes_reduced = boxes[keeps]

  offset = torch.zeros_like(objs)
  for i in range(objs.shape[0]):
    for j in idx:
      if j < i:
        offset[i] += 1

  # edges connected to removed object
  keeps_t = []
  triples_reduced = []
  for i in range(triples.shape[0]):
    if not(triples[i,0] in idx or triples[i, 2] in idx):
      keeps_t.append(i)
      triples_reduced.append(torch.tensor([triples[i,0] - offset[triples[i,0]], triples[i,1],
                                           triples[i,2] - offset[triples[i,2]]], device=triples.device))
  triples_reduced = torch.stack(triples_reduced, dim=0)

  # update indexing arrays
  obj_to_img_reduced = obj_to_img[keeps]
  triple_to_img_reduced = triple_to_img[keeps_t]

  # mask RoI of removed objects from image
  for i in idx:
    imgs = mask_image_in_bbox(imgs, boxes, i, obj_to_img, mode='removal')

  return objs_reduced, triples_reduced, boxes_reduced, imgs, obj_to_img_reduced, triple_to_img_reduced


def bbox_coordinates_with_margin(bbox, margin, img):
    # extract bounding box with a margin

    left = max(0, bbox[0] * img.shape[3] - margin)
    top = max(0, bbox[1] * img.shape[2] - margin)
    right = min(img.shape[3], bbox[2] * img.shape[3] + margin)
    bottom = min(img.shape[2], bbox[3] * img.shape[2] + margin)

    return int(left), int(right), int(top), int(bottom)


def save_image_from_tensor(img, img_dir, filename):

    img = imagenet_deprocess_batch(img)
    img_np = img[0].numpy().transpose(1, 2, 0)
    img_path = os.path.join(img_dir, filename)
    imsave(img_path, img_np)


def save_image_with_label(img_pred, img_gt, img_dir, filename, txt_str):
    # saves gt and generated image, concatenated
    # together with text label describing the change
    # used for easier visualization of results

    img_pred = imagenet_deprocess_batch(img_pred)
    img_gt = imagenet_deprocess_batch(img_gt)

    img_pred_np = img_pred[0].numpy().transpose(1, 2, 0)
    img_gt_np = img_gt[0].numpy().transpose(1, 2, 0)

    img_pred_np = cv2.resize(img_pred_np, (128, 128))
    img_gt_np = cv2.resize(img_gt_np, (128, 128))

    wspace = np.zeros([img_pred_np.shape[0], 10, 3])
    text = np.zeros([30, img_pred_np.shape[1] * 2 + 10, 3])
    text = cv2.putText(text, txt_str, (0,20), cv2.FONT_HERSHEY_SIMPLEX,
                     0.5, (255, 255, 255), lineType=cv2.LINE_AA)

    img_pred_gt = np.concatenate([img_gt_np, wspace, img_pred_np], axis=1).astype('uint8')
    img_pred_gt = np.concatenate([text, img_pred_gt], axis=0).astype('uint8')
    img_path = os.path.join(img_dir, filename)
    imsave(img_path, img_pred_gt)


def makedir(base, name, flag=True):
    dir_name = None
    if flag:
        dir_name = os.path.join(base, name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
    return dir_name


def save_graph_json(objs, triples, boxes, beforeafter, dir, idx):
    # save scene graph in json form

    data = {}
    objs = objs.cpu().numpy()
    triples = triples.cpu().numpy()
    data['objs'] = objs.tolist()
    data['triples'] = triples.tolist()
    data['boxes'] = boxes.tolist()
    with open(dir + '/' + beforeafter + '_' + str(idx) + '.json', 'w') as outfile:
        json.dump(data, outfile)


def query_image_by_semantic_id(obj_id, curr_img_id, loader, num_samples=7):
    # used to replace objects with an object of the same category and different appearance
    # return list of images and bboxes, that contain object of category obj_id

    query_imgs, query_boxes = [], []
    loader_id = 0
    counter = 0

    for l in loader:
        # load images
        imgs, objs, boxes, _, _, _, _ = [x.cuda() for x in l]
        if loader_id == curr_img_id:
            loader_id += 1
            continue

        for i, ob in enumerate(objs):
            if obj_id[0] == ob:
                query_imgs.append(imgs)
                query_boxes.append(boxes[i])
                counter += 1
            if counter == num_samples:
                return query_imgs, query_boxes
        loader_id += 1

    return 0, 0


def draw_image_box(img, box, color='g'):

    left, right, top, bottom = int(round(box[0] * img.shape[1])), int(round(box[2] * img.shape[1])), \
                               int(round(box[1] * img.shape[0])), int(round(box[3] * img.shape[0]))
    if color == 'r':
        color = (255,0,0)
    elif color == 'g':
        color = (0,255,0)
    cv2.rectangle(img, (left, top), (right, bottom), color, 1)
    return img


def draw_image_edge(img, box1, box2):
    # draw arrow that connects two objects centroids
    left1, right1, top1, bottom1 = int(round(box1[0] * img.shape[1])), int(round(box1[2] * img.shape[1])), \
                               int(round(box1[1] * img.shape[0])), int(round(box1[3] * img.shape[0]))
    left2, right2, top2, bottom2 = int(round(box2[0] * img.shape[1])), int(round(box2[2] * img.shape[1])), \
                               int(round(box2[1] * img.shape[0])), int(round(box2[3] * img.shape[0]))

    cv2.arrowedLine(img, (int((left1+right1)/2), int((top1+bottom1)/2)),
             (int((left2+right2)/2), int((top2+bottom2)/2)), (255,0,0), 1)

    return img


def visualize_imgs_boxes(imgs, imgs_pred, boxes, boxes_pred):

    nrows = imgs.size(0)
    imgs = imgs.detach().cpu().numpy()
    imgs_pred = imgs_pred.detach().cpu().numpy()
    boxes = boxes.detach().cpu().numpy()
    boxes_pred = boxes_pred.detach().cpu().numpy()
    plt.figure()

    for i in range(0, nrows):
        # i = j//2
        ax1 = plt.subplot(2, nrows, i+1)
        img = np.transpose(imgs[i, :, :, :], (1, 2, 0)) / 255.
        plt.imshow(img)

        left, right, top, bottom = bbox_coordinates_with_margin(boxes[i, :], 0, imgs[i:i+1, :, :, :])
        bbox_gt = patches.Rectangle((left, top),
                                    width=right-left,
                                    height=bottom-top,
                                    linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax1.add_patch(bbox_gt)
        plt.axis('off')

        ax2 = plt.subplot(2, nrows, i+nrows+1)
        pred = np.transpose(imgs_pred[i, :, :, :], (1, 2, 0)) / 255.
        plt.imshow(pred)

        left, right, top, bottom = bbox_coordinates_with_margin(boxes_pred[i, :], 0, imgs[i:i+1, :, :, :])
        bbox_pr = patches.Rectangle((left, top),
                                    width=right-left,
                                    height=bottom-top,
                                    linewidth=1, edgecolor='r', facecolor='none')
        # ax2.add_patch(bbox_gt)
        ax2.add_patch(bbox_pr)
        plt.axis('off')

    plt.show()


def visualize_scene_graphs(args, obj_to_img, objs, triples, vocab, device, img_idx=0, indirect_mode=None, modified_obj_pos=None):
    ############### save path ##################
    if args.random_feats:
       image_task = 'image_synthesis'
    else:
       image_task = 'image_reconstruction'
      

    subdir = os.path.join('unlearning',
                          image_task,
                          str(args.specific_obj)+'_'+str(args.data_lot_idx))
    
    IMG_SAVE_PATH_COMBINED = args.image_save_path + subdir

    if not os.path.exists(IMG_SAVE_PATH_COMBINED):
       os.makedirs(IMG_SAVE_PATH_COMBINED)
    
    ############### scene graph visualization ##################
    offset = 0
    for i in range(1):#imgs_in.size(0)):
        curr_obj_idx = (obj_to_img == i).nonzero()

        objs_vis = objs[curr_obj_idx]
        triples_vis = []
        for j in range(triples.size(0)):
            if triples[j, 0] in curr_obj_idx or triples[j, 2] in curr_obj_idx:
                triples_vis.append(triples[j].to(device) - torch.tensor([offset, 0, offset]).to(device))
        offset += curr_obj_idx.size(0)
        triples_vis = torch.stack(triples_vis, 0)

        # print(objs_vis, triples_vis)

        if img_idx == args.candidate:
            specific_object = args.unlearn_obj_and_idx
        else:
            specific_object = None
        
        
        
        # cv2.imshow('graph' + str(i), graph_img)
        if indirect_mode == None:
            graph_img_TB = draw_scene_graph(objs_vis, triples_vis, vocab, specific_object, orientation='TB')
            graph_img_RL = draw_scene_graph(objs_vis, triples_vis, vocab, specific_object, orientation='RL')
            imsave(IMG_SAVE_PATH_COMBINED + f'/{img_idx}_scene_graph_TB.png', graph_img_TB)
            imsave(IMG_SAVE_PATH_COMBINED + f'/{img_idx}_scene_graph_RL.png', graph_img_RL)
        else:
            graph_img_TB = draw_scene_graph(objs_vis, triples_vis, vocab, specific_object, orientation='TB', modified_object_pos=modified_obj_pos)
            graph_img_RL = draw_scene_graph(objs_vis, triples_vis, vocab, specific_object, orientation='RL', modified_object_pos=modified_obj_pos)
            imsave(IMG_SAVE_PATH_COMBINED + f'/{img_idx}_scene_graph_TB_{indirect_mode}.png', graph_img_TB)
            imsave(IMG_SAVE_PATH_COMBINED + f'/{img_idx}_scene_graph_RL_{indirect_mode}.png', graph_img_RL)



    cv2.waitKey(10000)


def remove_duplicates(triples, triple_to_img, indexes):
    # removes duplicates in relationship triples

    triples_new = []
    triple_to_img_new = []

    for i in range(triples.size(0)):
        if i not in indexes:
            triples_new.append(triples[i])
            triple_to_img_new.append(triple_to_img[i])

    triples_new = torch.stack(triples_new, 0)
    triple_to_img_new = torch.stack(triple_to_img_new, 0)

    return triples_new, triple_to_img_new


def parse_bool(pred_graphs, generative, use_gt_boxes, use_feats):
    # returns name of output directory depending on arguments

    if pred_graphs:
        name = "pred/"
    else:
        name = ""
    if generative: # fully generative mode
        return name + "generative"
    else:
        if use_gt_boxes:
            b = "withbox"
        else:
            b = "nobox"
        if use_feats:
            f = "withfeats"
        else:
            f = "nofeats"

        return name + b + "_" + f




def is_background(label_id):

    if label_id in [169, 60, 61, 49, 141, 8, 11, 52, 66]:
        return True
    else:
        return False


def get_selected_objects():

  objs = ["", "apple", "ball", "banana", "beach", "bike", "bird", "bus", "bush", "cat", "car", "chair", "cloud", "dog",
          "elephant", "field", "giraffe", "man", "motorcycle", "ocean", "person", "plane", "sheep", "tree", "zebra"]

  return objs


def save_images(args, candidate_, all_imgs, img_idx, boxes, box_on=False, all_data=None, first_only=True, ablation_mode=None, indirect_mode=None, indirect_img=None):

    if args.random_feats:
       image_task = 'image_synthesis'
    else:
       image_task = 'image_reconstruction'
    
    # ablation_model is for ablation study on influence function-based method.
    if ablation_mode == "scalar":
        subdir = os.path.join('unlearning',
                          image_task,
                          'ablation',
                          'scalar',
                          str(args.scale)
                          )
    elif ablation_mode == "module":
        subdir = os.path.join('unlearning',
                          image_task,
                          'ablation',
                          'module',
                          args.unlearn_module
                          )
    else:
        subdir = os.path.join('unlearning',
                            image_task,
                            str(args.specific_obj)+'_'+str(args.data_lot_idx))

    if not img_idx == args.candidate:
        sample = all_data[img_idx][0]
        obj_indices = [index for index, obj_class in enumerate(sample[1]) if obj_class == args.specific_obj]
        obj = [int(sample[1][idx]) for idx in obj_indices]
        same_obj = (obj[0], obj_indices[0])
        
        
    
    IMG_SAVE_PATH_COMBINED = args.image_save_path + subdir

    if not os.path.exists(IMG_SAVE_PATH_COMBINED):
       os.makedirs(IMG_SAVE_PATH_COMBINED)


    ################### Image post-processing ###################
    if indirect_mode == None:
        for i in range(len(all_imgs)):
            all_imgs[i] = all_imgs[i].detach().cpu().numpy().transpose([0, 2, 3, 1])
            
            if box_on: # Draw BBox of unlearned object
                if img_idx == args.candidate:
                    all_imgs[i][0] = draw_image_box(all_imgs[i][0].copy(), boxes[args.unlearn_obj_and_idx[1]].cpu().numpy(), color='g') # draw requested object
                else:
                    all_imgs[i][0] = draw_image_box(all_imgs[i][0].copy(), boxes[same_obj[1]].cpu().numpy(), color='r') # draw same category of object in other samples
    else:
        indirect_img = indirect_img.detach().cpu().numpy().transpose([0, 2, 3, 1])
        if box_on: # Draw BBox of unlearned object
            indirect_img[0] = draw_image_box(indirect_img[0].copy(), boxes[args.unlearn_obj_and_idx[1]].cpu().numpy(), color='g') # draw requested object
    


    if first_only:
        # print(candidate_)
        print(IMG_SAVE_PATH_COMBINED)
        if not os.path.exists(IMG_SAVE_PATH_COMBINED + '/gt'):
           os.makedirs(IMG_SAVE_PATH_COMBINED + '/gt')
        if not os.path.exists(IMG_SAVE_PATH_COMBINED + '/gen'):
           os.makedirs(IMG_SAVE_PATH_COMBINED + '/gen')
        if indirect_mode == None:
            for method in args.methods_involved_list:
                if not os.path.exists(IMG_SAVE_PATH_COMBINED + f'/{method}'):
                    os.makedirs(IMG_SAVE_PATH_COMBINED + f'/{method}')
        else:
            for method in args.methods_involved_list:
                if not os.path.exists(IMG_SAVE_PATH_COMBINED + f'/{method}' + f'/{indirect_mode}'):
                    os.makedirs(IMG_SAVE_PATH_COMBINED + f'/{method}' + f'/{indirect_mode}')
            

        
        
        if img_idx == candidate_:
            candidate_ = f'{img_idx}_unl'
        else:
            candidate_ = str(img_idx)
        
        if indirect_mode == None:
            imsave(IMG_SAVE_PATH_COMBINED + '/gt/' + str(candidate_) + '.png', all_imgs[0][0].astype('uint8'))
            imsave(IMG_SAVE_PATH_COMBINED + '/gen/' + str(candidate_) + '.png', all_imgs[1][0].astype('uint8'))
            for index, method in enumerate(args.methods_involved_list):
                imsave(IMG_SAVE_PATH_COMBINED + f'/{method}/' + str(candidate_) + '.png', all_imgs[index+2][0].astype('uint8'))
        else:
            for method in args.methods_involved_list:
                imsave(IMG_SAVE_PATH_COMBINED + f'/{method}/' + f'/{indirect_mode}/' + str(candidate_) + '.png', indirect_img[0].astype('uint8'))


        
            


########################## Monkey Patch for F.grid_sample ###########################
def custom_grid_sample(image, optical):
    '''
    https://github.com/pytorch/pytorch/issues/34704 AliaksandrSiarohin's answer to :
    “RuntimeError: derivative for grid_sampler_2d_backward is not implemented”
    '''
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

def compare_models(model1, model2):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(param1, param2):
            return False
    return True

def compare_complex_outputs(output1, output2):
    for elem1, elem2 in zip(output1, output2):
        if isinstance(elem1, torch.Tensor) and isinstance(elem2, torch.Tensor):
            if not torch.equal(elem1, elem2):
                print('不等')
                return False
            if elem1 != elem2:
                print('不等')
                return False

    return True

def tensor2image(tensor):
  # if tensor.ndimension == 4:
  #     tensor = tensor.squeeze(0)
  # image_array = tensor.permute(1, 2, 0).numpy()
  # image_array = (image_array * 255).astype(np.uint8)
  # image = Image.fromarray(image_array)
  # image.save('tmp/tmp_iamge.png')
  # print(1)
  unloader = transforms.ToPILImage()
  image = tensor.cpu().clone()  # clone the tensor to not modify the original tensor
  image = image.squeeze(0)      # remove the fake batch dimension
  image = unloader(image)
  image.save('tmp/tmp_iamge.png')






