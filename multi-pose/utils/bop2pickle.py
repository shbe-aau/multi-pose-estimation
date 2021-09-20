# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Visualizes object models in the ground-truth poses."""

import os
import numpy as np
import argparse
import pickle
import cv2

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc

# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': 'tless',

  # Dataset split. Options: 'train', 'val', 'test'.
  'dataset_split': 'train',

  # Dataset split type. None = default. See dataset_params.py for options.
  'dataset_split_type': None,

  # File with a list of estimation targets used to determine the set of images
  # for which the GT poses will be visualized. The file is assumed to be stored
  # in the dataset folder. None = all images.
  # 'targets_filename': 'test_targets_bop19.json',
  'targets_filename': None,

  # Select ID's of scenes, images and GT poses to be processed.
  # Empty list [] means that all ID's will be used.
  'scene_ids': [],
  'im_ids': [], #1,101,201,301,401,501],
  'gt_ids': [],
  'obj_ids': [19],

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,
}
################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', default=p['datasets_path'])
parser.add_argument('--dataset_split', default=p['dataset_split'])
parser.add_argument('--dataset', default=p['dataset'])
parser.add_argument('--scene_ids', type=int, nargs='+', default=p['scene_ids'])
parser.add_argument('--obj_ids', type=int, nargs='+', default=p['obj_ids'])
args = parser.parse_args()

p['datasets_path'] = str(args.datasets_path)
p['dataset_split'] = str(args.dataset_split)
p['dataset'] = str(args.dataset)
p['scene_ids'] = args.scene_ids
p['obj_ids'] = args.obj_ids

visualize = False

def extract_square_patch(scene_img, bb_xywh, pad_factor=1.2,resize=(128,128),
                         interpolation=cv2.INTER_NEAREST,black_borders=False):

        x, y, w, h = np.array(bb_xywh).astype(np.int32)
        size = int(np.maximum(h, w) * pad_factor)

        left = int(np.maximum(x+w/2-size/2, 0))
        right = int(np.minimum(x+w/2+size/2, scene_img.shape[1]))
        top = int(np.maximum(y+h/2-size/2, 0))
        bottom = int(np.minimum(y+h/2+size/2, scene_img.shape[0]))

        scene_crop = scene_img[top:bottom, left:right].copy()

        if black_borders:
            scene_crop[:(y-top),:] = 0
            scene_crop[(y+h-top):,:] = 0
            scene_crop[:,:(x-left)] = 0
            scene_crop[:,(x+w-left):] = 0

        scene_crop = cv2.resize(scene_crop, resize) #, interpolation = interpolation)
        return scene_crop

# Load dataset parameters.
dp_split = dataset_params.get_split_params(
  p['datasets_path'], p['dataset'], p['dataset_split'], p['dataset_split_type'])

model_type = 'eval'  # None = default.
dp_model = dataset_params.get_model_params(
  p['datasets_path'], p['dataset'], model_type)

# Subset of images for which the ground-truth poses will be rendered.
if p['targets_filename'] is not None:
  targets = inout.load_json(
    os.path.join(dp_split['base_path'], p['targets_filename']))
  scene_im_ids = {}
  for target in targets:
    scene_im_ids.setdefault(
      target['scene_id'], set()).add(target['im_id'])
else:
  scene_im_ids = None

# List of considered scenes.
scene_ids_curr = dp_split['scene_ids']
if p['scene_ids']:
  scene_ids_curr = set(scene_ids_curr).intersection(p['scene_ids'])

# Save scene data in a dict
data={"images":[],
      "Rs":[],
      "ts":[],
      "visib_fract":[],
      "scene_ids":[],
      "img_ids":[],
      "obj_ids":[]}

for scene_id in scene_ids_curr:
  # Load scene info and ground-truth poses.
  #scene_camera = inout.load_scene_camera(
  #  dp_split['scene_camera_tpath'].format(scene_id=scene_id))
  scene_gt = inout.load_scene_gt(
    dp_split['scene_gt_tpath'].format(scene_id=scene_id))
  scene_gt_info = inout.load_scene_gt(
    dp_split['scene_gt_info_tpath'].format(scene_id=scene_id))

  # List of considered images.
  if scene_im_ids is not None:
    im_ids = scene_im_ids[scene_id]
  else:
    im_ids = sorted(scene_gt.keys())
  if p['im_ids']:
    im_ids = set(im_ids).intersection(p['im_ids'])

  # Loops through the images
  for im_counter, im_id in enumerate(im_ids):
    # List of considered ground-truth poses.
    gt_ids_curr = range(len(scene_gt[im_id]))
    if p['gt_ids']:
      gt_ids_curr = set(gt_ids_curr).intersection(p['gt_ids'])

    # Collect the ground-truth poses.
    gt_poses = []
    for gt_id in gt_ids_curr:
      gt = scene_gt[im_id][gt_id]
      gt_info = scene_gt_info[im_id][gt_id]

      # Load info
      obj_id = gt['obj_id']
      bbox = gt_info['bbox_obj']
      #bbox = gt_info['bbox_visib']
      visible = gt_info['visib_fract']

      if(len(p['obj_ids']) > 0 and obj_id not in p['obj_ids']):
        continue

      # Load the color image
      rgb = inout.load_im(dp_split['rgb_tpath'].format(scene_id=scene_id, im_id=im_id))[:, :, :3]

      bb = gt_info['bbox_visib']

      if(bb[2] == 0 or bb[3] == 0):
        continue
      cropped = extract_square_patch(rgb, bb)

      data["images"].append(cropped)
      data["Rs"].append(gt['cam_R_m2c'])
      data["ts"].append(gt['cam_t_m2c'])
      data["visib_fract"].append(gt_info['visib_fract'])
      data["scene_ids"].append(scene_id)
      data["img_ids"].append(im_id)
      data["obj_ids"].append(gt['obj_id'])

      print("Processing image {0}/{1}".format(im_counter+1,len(im_ids)))

      if(visualize):
        # Draw bbox and display
        color = (0,255,0)
        if(visible < 0.2):
          color = (255,0,0)
        vis_rgb = cv2.rectangle(rgb.copy(),
                                (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), color, 2)
        cv2.imshow("scene", np.flip(vis_rgb,axis=2))
        cv2.imshow("object crop", np.flip(cropped,axis=2))
        cv2.waitKey(0)

  if(len(data["images"]) == 0):
    continue

if(p["dataset_split"] == "train"):
  output_path = "./train-obj{0:02d}.p".format(data["obj_ids"][0])
else:
  output_path = "./test-obj{0:02d}.p".format(data["obj_ids"][0])
print(output_path)
pickle.dump(data, open(output_path, "wb"), protocol=2)
