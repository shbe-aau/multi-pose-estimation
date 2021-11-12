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

    bg_path = "../../autoencoder_ws/data/VOC2012/JPEGImages/"
    datagen = DatasetGenerator(args.get('Dataset', 'BACKGROUND_IMAGES'),
                                     model_path_data,
                                     translations,
                                     args.getint('Training', 'BATCH_SIZE'),
                                     "not_used",
                                     device,
                                     sampling_method = args.get('Training', 'VIEW_SAMPLING'),
                                     max_rel_offset = args.getfloat('Training', 'MAX_REL_OFFSET', fallback=0.2),
                                     augment_imgs = args.getboolean('Training', 'AUGMENT_IMGS', fallback=True),
                                     seed=args.getint('Training', 'RANDOM_SEED'))

    output_path = args.get('Training', 'OUTPUT_PATH')

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


    # make training data
    datagen.max_samples = args.getint('Training', 'NUM_SAMPLES')
    last_num = gen_data(datagen, "train.txt", data_dir)

    # make validation data
    datagen.max_samples = args.getint('Training', 'NUM_SAMPLES')
    gen_data(datagen, "valid.txt", data_dir, last_num)



def gen_data(datagen, name, data_dir, last_num=0):
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
        for curr_batch in datagen:
            for i in range(len(curr_batch["images"])):
                ind += 1
                # save congregate image to images folder as *ind*.png or .jpg?
                im_file = os.path.join(im_path, "{}.png".format(ind))
                result=cv.imwrite(im_file, curr_batch["images"][i]*255)
                if result==False:
                    print("Error in saving image {}".format(ind))

                # save bbox of each object in image as:
                # label_idx x_center y_center width height
                # label_idx = (objectnum-1 as it is 0 indexed in classes.names)
                # [0, 1] scaled coordinates

                # write image name as row in imlistfile to put in correct subset
                imlistfile.write("{}\n".format(im_file))

        return ind


if __name__ == '__main__':
    main()
