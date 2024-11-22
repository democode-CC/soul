from sg2i.data.vg import SceneGraphNoPairsDataset, collate_fn_nopairs
from sg2i.data.clevr import SceneGraphWithPairsDataset, collate_fn_withpairs

import json
from torch.utils.data import DataLoader, Subset


def build_clevr_supervised_train_dsets(args):
  print("building fully supervised %s dataset" % args.dataset)
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.train_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_train_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
  }
  train_dset = SceneGraphWithPairsDataset(**dset_kwargs)
  iter_per_epoch = len(train_dset) // args.batch_size
  print('There are %d iterations per epoch' % iter_per_epoch)

  dset_kwargs['h5_path'] = args.val_h5
  del dset_kwargs['max_samples']
  val_dset = SceneGraphWithPairsDataset(**dset_kwargs)

  dset_kwargs['h5_path'] = args.test_h5
  test_dset = SceneGraphWithPairsDataset(**dset_kwargs)

  return vocab, train_dset, val_dset, test_dset


def build_dset_nopairs(args, checkpoint):

  vocab = checkpoint['model_kwargs']['vocab']
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.data_h5,
    'image_dir': args.data_image_dir,
    'image_size': args.image_size,
    'max_objects': checkpoint['args']['max_objects_per_image'],
    'use_orphaned_objects': checkpoint['args']['vg_use_orphaned_objects'],
    'mode': args.mode,
    'predgraphs': args.predgraphs
  }
  dset = SceneGraphNoPairsDataset(**dset_kwargs)

  return dset


def build_dset_withpairs(args, checkpoint, vocab_t):

  vocab = vocab_t
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.data_h5,
    'image_dir': args.data_image_dir,
    'image_size': args.image_size,
    'max_objects': checkpoint['args']['max_objects_per_image'],
    'use_orphaned_objects': checkpoint['args']['vg_use_orphaned_objects'],
    'mode': args.mode
  }
  dset = SceneGraphWithPairsDataset(**dset_kwargs)

  return dset


def build_eval_loader(args, checkpoint, vocab_t=None, no_gt=False):

  if args.dataset == 'vg' or (no_gt and args.dataset == 'clevr'):
    dset = build_dset_nopairs(args, checkpoint)
    collate_fn = collate_fn_nopairs
  elif args.dataset == 'clevr':
    dset = build_dset_withpairs(args, checkpoint, vocab_t)
    collate_fn = collate_fn_withpairs

  loader_kwargs = {
    'batch_size': 1,
    'num_workers': args.loader_num_workers,
    'shuffle': args.shuffle,
    'collate_fn': collate_fn,
  }
  loader = DataLoader(dset, **loader_kwargs)

  return loader


def build_train_dsets(args):
  print("building unpaired %s dataset" % args.dataset)
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.train_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_train_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
    'finetune':args.finetune,
  }
  train_dset = SceneGraphNoPairsDataset(**dset_kwargs)
  iter_per_epoch = len(train_dset) // args.batch_size
  print('There are %d iterations per epoch' % iter_per_epoch)

  dset_kwargs['h5_path'] = args.val_h5
  del dset_kwargs['max_samples']
  val_dset = SceneGraphNoPairsDataset(**dset_kwargs)

  return vocab, train_dset, val_dset

def sample_specific_data_for_ojb_unl(args, trainset, specific_obj, obj_size_threshold=0.3):
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
  
  

  selected_indices = []
  selected_count = 0
  for idx, batch in enumerate(trainset):
    if (specific_obj in batch[1]):
      obj_index = list(batch[1]).index(specific_obj)
      left, right = batch[2][obj_index][0], batch[2][obj_index][2]
      width = right - left 
      if width > obj_size_threshold:
        if selected_count >= args.data_lot_idx:
          selected_indices.append(idx)
        selected_count += 1
    
    if len(selected_indices) >= args.trainset_size:
      break

  return selected_indices

def build_train_loaders(args):

  print(args.dataset)
  if args.dataset == 'vg' or (args.dataset == "clevr" and not args.is_supervised):
    vocab, train_dset, val_dset = build_train_dsets(args) # a bit slow
    collate_fn = collate_fn_nopairs
  elif args.dataset == 'clevr':
    vocab, train_dset, val_dset, test_dset = build_clevr_supervised_train_dsets(args)
    collate_fn = collate_fn_withpairs

  

  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': args.shuffle_train_loader,
    'collate_fn': collate_fn,
  }

  if not args.specific_obj == None:
    selected_indices = sample_specific_data_for_ojb_unl(args, train_dset, args.specific_obj, obj_size_threshold=args.obj_size_threshold)
    train_loader = DataLoader(Subset(train_dset, selected_indices), **loader_kwargs)
    check_loader(train_loader)
  else:
    train_loader = DataLoader(train_dset, **loader_kwargs)



  

  loader_kwargs['shuffle'] = args.shuffle_val
  val_loader = DataLoader(val_dset, **loader_kwargs)

  print("shuffle_train_loader", args.shuffle_train_loader)

  return vocab, train_loader, val_loader


def check_loader(train_loader):
  for batch in train_loader:
    assert len(batch) == 7
  print('pass new loader check!')