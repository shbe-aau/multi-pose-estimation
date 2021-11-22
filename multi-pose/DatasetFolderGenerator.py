import os
import argparse
import configparser
import json
import imgaug as ia
import torch
import numpy as np
import random
import cv2 as cv
from DatasetGeneratorOpenGL import DatasetGenerator
from utils.utils import *
from utils.tools import *

def main():
    # Read configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    arguments = parser.parse_args()

    cfg_file_path = os.path.join("./experiments", arguments.experiment_name)
    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    seed=args.getint('Training', 'RANDOM_SEED')
    if(seed is not None):
        torch.manual_seed(seed)
        #torch.use_deterministic_algorithms(True) # Requires pytorch>=1.8.0
        #torch.backends.cudnn.deterministic = True
        np.random.seed(seed=seed)
        ia.seed(seed)
        random.seed(seed)

    # Set the cuda device
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Handle loading of multiple object paths and translations
    try:
        model_path_data = json.loads(args.get('Dataset', 'MODEL_PATH_DATA'))
        translations = np.array(json.loads(args.get('Rendering', 'T')))
    except:
        model_path_data = [args.get('Dataset', 'MODEL_PATH_DATA')]
        translations = [np.array(json.loads(args.get('Rendering', 'T')))]

    camera_params = calc_camera_parameters(render_width=args.getint('Rendering', 'RENDER_WIDTH',fallback=400),
                                           render_height=args.getint('Rendering', 'RENDER_HEIGHT', fallback=400),
                                           orig_width=args.getint('Rendering', 'ORIGINAL_WIDTH', fallback=720),
                                           orig_height=args.getint('Rendering', 'ORIGINAL_HEIGHT', fallback=540),
                                           fx=args.getfloat('Rendering', 'FOCAL_X', fallback=1075.65091572),
                                           fy=args.getfloat('Rendering', 'FOCAL_Y', fallback=1073.90347929))

    #bg_path = "../../autoencoder_ws/data/VOC2012/JPEGImages/"
    bg_path = "/home/hampus/vision/PyTorch-YOLOv3/data/coco/images/val2014"
    datagen = DatasetGenerator(args.get('Dataset', 'BACKGROUND_IMAGES'),
                                     model_path_data,
                                     translations,
                                     args.getint('Training', 'BATCH_SIZE'),
                                     "not_used",
                                     device,
                                     camera_params,
                                     sampling_method = args.get('Training', 'VIEW_SAMPLING'),
                                     max_rel_offset = args.getfloat('Training', 'MAX_REL_OFFSET', fallback=0.2),
                                     augment_imgs = args.getboolean('Training', 'AUGMENT_IMGS', fallback=True),
                                     seed=args.getint('Training', 'RANDOM_SEED'),
                                     gen_scenes=True)

    data_dir = args.get('Datagen', 'OUTPUT_PATH', fallback=os.path.join(os.getcwd(), "detection_data"))

    classes = list(range(1, 31))

    # prep main directory
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # save classes
    path = os.path.join(data_dir, "classes.names")
    with open(path, "w") as file:
        for cls in classes:
            file.write("{}\n".format(cls))

    width = camera_params['render_width']
    height = camera_params['render_height']

    # make training data
    datagen.max_samples = args.getint('YOLO', 'TRAIN_SAMPLES')
    last_num = gen_data(datagen, "train.txt", data_dir, width, height)

    # make validation data
    datagen.max_samples = args.getint('YOLO', 'VAL_SAMPLES')
    gen_data(datagen, "valid.txt", data_dir, width, height, last_num)

def bb_reformat(bboxin, render_width, render_height):
    tlx, tly, w, h = bboxin

    cx = tlx + w/2
    cy = tly + h/2
    return [cx/render_width, cy/render_height, w/render_width, w/render_height]

def gen_data(datagen, name, data_dir, width, height, last_num=0):
    # Create directory
    #path = os.path.join(data_dir, name)
    #os.mkdir(path)

    im_path = os.path.join(data_dir, "images")
    if not os.path.isdir(im_path):
        os.mkdir(im_path)
    label_path = os.path.join(data_dir, "labels")
    if not os.path.isdir(label_path):
        os.mkdir(label_path)

    path = os.path.join(data_dir, name)
    with open(path, "w") as imlistfile:
        ind = last_num
        for scene_image, obj_dict in datagen:
            ind += 1
            # save congregate image to images folder as *ind*.png or .jpg?
            im_file = os.path.join(im_path, "{}.png".format(ind))
            result=cv.imwrite(im_file, scene_image*255)
            if result==False:
                print("Error in saving image {}".format(ind))

            # save bbox of each object in image as:
            # label_idx x_center y_center width height
            # label_idx = (objectnum-1 as it is 0 indexed in classes.names)
            # [0, 1] scaled coordinates
            label_path_txt = os.path.join(label_path, "{}.txt".format(ind))
            with open(label_path_txt, "w") as label_file:
                for object in obj_dict:
                    cx, cy, w, h = bb_reformat(object["bbox"], width, height)
                    label_file.write("{} {} {} {} {}\n".format(object["obj_id"], cx, cy, w, h))

            # write image name as row in imlistfile to put in correct subset
            imlistfile.write("{}\n".format(im_file))

        return ind


if __name__ == '__main__':
    main()
